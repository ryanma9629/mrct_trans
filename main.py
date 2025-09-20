import os
import csv
import re
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    llm_provider: str
    api_token: str

class TranslationResponse(BaseModel):
    translated_text: str
    dictionary_matches: List[Tuple[str, str, int, int]]

class DictionaryMatcher:
    def __init__(self, csv_file: str):
        self.en_to_cn = {}
        self.cn_to_en = {}
        self.load_dictionary(csv_file)
    
    def load_dictionary(self, csv_file: str):
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    en_term = row[0].strip()
                    cn_term = row[1].strip()
                    self.en_to_cn[en_term.lower()] = cn_term
                    self.cn_to_en[cn_term] = en_term
    
    def find_matches(self, text: str, is_english: bool) -> List[Tuple[str, str, int, int]]:
        matches = []
        dictionary = self.en_to_cn if is_english else self.cn_to_en
        
        # Sort terms by length (longest first) to handle overlap
        sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)
        
        text_lower = text.lower() if is_english else text
        used_positions = set()
        
        for term in sorted_terms:
            if is_english:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(term))
            
            for match in pattern.finditer(text_lower if is_english else text):
                start, end = match.span()
                # Check if this position is already used
                if not any(pos in used_positions for pos in range(start, end)):
                    matches.append((term, dictionary[term], start, end))
                    used_positions.update(range(start, end))
        
        return sorted(matches, key=lambda x: x[2])  # Sort by start position

dictionary_matcher = DictionaryMatcher("QS-TB.csv")

async def call_chatgpt(text: str, api_token: str, is_english: bool, dictionary_matches: List[Tuple[str, str, int, int]] = None, is_azure: bool = False) -> str:
    direction = "from English to Chinese" if is_english else "from Chinese to English"
    
    # Prepare dictionary context for the prompt
    dict_context = ""
    if dictionary_matches:
        dict_context = "\n\nIMPORTANT: Use these specific dictionary translations when they appear in the text:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nUse these exact translations for the matching terms, but translate the rest of the text normally."
    
    system_prompt = f"You are a translation service. Your only task is to translate text {direction}. Do not chat, do not explain, do not add greetings. Just provide the direct translation of the input text.{dict_context}"
    
    if is_azure:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not configured")
        
        client = AsyncAzureOpenAI(
            api_key=api_token,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        )
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"}
            ],
            max_tokens=2000
        )
    else:
        client = AsyncOpenAI(api_key=api_token)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"}
            ],
            max_tokens=2000
        )
    
    return response.choices[0].message.content.strip()

async def call_deepseek(text: str, api_token: str, is_english: bool, dictionary_matches: List[Tuple[str, str, int, int]] = None) -> str:
    endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
    
    # Prepare dictionary context for the prompt
    dict_context = ""
    if dictionary_matches:
        dict_context = "\n\nIMPORTANT: Use these specific dictionary translations when they appear in the text:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nUse these exact translations for the matching terms, but translate the rest of the text normally."
    
    system_prompt = f"You are a translation service. Your only task is to translate text {'from English to Chinese' if is_english else 'from Chinese to English'}. Do not chat, do not explain, do not add greetings. Just provide the direct translation of the input text.{dict_context}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                "max_tokens": 4000
            }
        )
        
        if response.status_code != 200:
            error_detail = f"DeepSeek API error: {response.status_code}"
            try:
                error_body = response.json()
                if "error" in error_body:
                    error_detail += f" - {error_body['error'].get('message', 'Unknown error')}"
            except:
                error_detail += f" - Response: {response.text[:200]}"
            raise HTTPException(status_code=400, detail=error_detail)
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

async def call_qwen(text: str, api_token: str, is_english: bool, dictionary_matches: List[Tuple[str, str, int, int]] = None) -> str:
    endpoint = os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    
    # Prepare dictionary context for the prompt
    dict_context = ""
    if dictionary_matches:
        dict_context = "\n\nIMPORTANT: Use these specific dictionary translations when they appear in the text:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nUse these exact translations for the matching terms, but translate the rest of the text normally."
    
    system_prompt = f"You are a translation service. Your only task is to translate text {'from English to Chinese' if is_english else 'from Chinese to English'}. Do not chat, do not explain, do not add greetings. Just provide the direct translation of the input text.{dict_context}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen-turbo",
                "input": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate: {text}"}
                    ]
                },
                "parameters": {
                    "max_tokens": 2000
                }
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Qwen API error")
        
        result = response.json()
        return result["output"]["text"].strip()

def detect_language(text: str) -> bool:
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # If input is purely English or mostly English, return True
    # If input has Chinese characters (including mixed), return False to translate to English
    return len(chinese_chars) == 0 or len(chinese_chars) < len(text.replace(' ', '')) * 0.1


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        is_english = detect_language(request.text)
        
        # Find dictionary matches in original text
        matches = dictionary_matcher.find_matches(request.text, is_english)
        
        # Call the appropriate LLM with dictionary context
        if request.llm_provider == "chatgpt":
            translated = await call_chatgpt(request.text, request.api_token, is_english, matches, False)
        elif request.llm_provider == "chatgpt(azure)":
            translated = await call_chatgpt(request.text, request.api_token, is_english, matches, True)
        elif request.llm_provider == "deepseek":
            translated = await call_deepseek(request.text, request.api_token, is_english, matches)
        elif request.llm_provider == "qwen":
            translated = await call_qwen(request.text, request.api_token, is_english, matches)
        else:
            raise HTTPException(status_code=400, detail="Unsupported LLM provider")
        
        return TranslationResponse(
            translated_text=translated,
            dictionary_matches=matches
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    return {
        "default_tokens": {
            "chatgpt": os.getenv("OPENAI_API_KEY", ""),
            "chatgpt(azure)": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
            "qwen": os.getenv("QWEN_API_KEY", "")
        }
    }

@app.get("/download-dictionary")
async def download_dictionary():
    return FileResponse(
        path="QS-TB.csv",
        filename="QS-TB.csv",
        media_type="text/csv"
    )

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)