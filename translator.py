import os
import csv
import re
from typing import Dict, List, Tuple, Optional
from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()


class DictionaryMatcher:
    """Handles technical dictionary matching for translation context."""
    
    def __init__(self, csv_file: str):
        self.en_to_cn = {}
        self.cn_to_en = {}
        self.load_dictionary(csv_file)
    
    def load_dictionary(self, csv_file: str):
        """Load dictionary from CSV file."""
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    en_term = row[0].strip()
                    cn_term = row[1].strip()
                    self.en_to_cn[en_term.lower()] = cn_term
                    self.cn_to_en[cn_term] = en_term
    
    def find_matches(self, text: str, is_english: bool) -> List[Tuple[str, str, int, int]]:
        """Find dictionary matches in text."""
        matches = []
        dictionary = self.en_to_cn if is_english else self.cn_to_en
        
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
                if not any(pos in used_positions for pos in range(start, end)):
                    matches.append((term, dictionary[term], start, end))
                    used_positions.update(range(start, end))
        
        return sorted(matches, key=lambda x: x[2])


class TranslationService:
    """Main translation service handling all LLM providers and dictionary matching."""
    
    def __init__(self, dictionary_file: str = "QS-TB.csv"):
        self.dictionary_matcher = DictionaryMatcher(dictionary_file)
    
    def detect_language(self, text: str) -> bool:
        """Detect if text is English (True) or Chinese (False)."""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) == 0 or len(chinese_chars) < len(text.replace(' ', '')) * 0.1
    
    def _prepare_dictionary_context(self, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        """Prepare dictionary context for LLM prompt."""
        dict_context = ""
        if dictionary_matches:
            dict_context = "\n\nIMPORTANT: Use these specific dictionary translations when they appear in the text:\n"
            for original_term, translated_term, _, _ in dictionary_matches:
                dict_context += f"- '{original_term}' → '{translated_term}'\n"
            dict_context += "\nUse these exact translations for the matching terms, but translate the rest of the text normally."
        return dict_context
    
    def _prepare_system_prompt(self, is_english: bool, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        """Prepare the system prompt for translation with dictionary context."""
        direction = "from English to Chinese" if is_english else "from Chinese to English"
        dict_context = self._prepare_dictionary_context(dictionary_matches)
        
        return f"You are a translation service. Your only task is to translate text {direction}. Do not chat, do not explain, do not add greetings. Just provide the direct translation of the input text.{dict_context}"
    
    async def _call_llm_api(self, text: str, api_token: str, is_english: bool, 
                           dictionary_matches: Optional[List[Tuple[str, str, int, int]]], 
                           provider: str, model: Optional[str] = None) -> str:
        """Unified method to call any LLM provider API."""
        system_prompt = self._prepare_system_prompt(is_english, dictionary_matches)
        
        if provider == "chatgpt":
            return await self._call_openai_api(text, api_token, system_prompt, False, model)
        elif provider == "chatgpt(azure)":
            return await self._call_openai_api(text, api_token, system_prompt, True, model)
        elif provider == "deepseek":
            return await self._call_http_api(text, api_token, system_prompt, "deepseek", model)
        elif provider == "qwen":
            return await self._call_http_api(text, api_token, system_prompt, "qwen", model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    async def _call_openai_api(self, text: str, api_token: str, system_prompt: str, 
                              is_azure: bool, model: Optional[str]) -> str:
        """Call OpenAI or Azure OpenAI API."""
        if is_azure:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT not configured")
            
            client = AsyncAzureOpenAI(
                api_key=api_token,
                azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            )
            default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
            deployment_name = model if model else default_model
            
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
            default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
            selected_model = model if model else default_model
            
            response = await client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                max_tokens=2000
            )
        
        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    
    async def _call_http_api(self, text: str, api_token: str, system_prompt: str, 
                            provider: str, model: Optional[str]) -> str:
        """Call HTTP-based API providers (DeepSeek, Qwen)."""
        if provider == "deepseek":
            endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/chat/completions")
            default_model = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
            timeout = 60.0
            max_tokens = 4000
        elif provider == "qwen":
            endpoint = os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            default_model = os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo")
            timeout = None
            max_tokens = 2000
        else:
            raise ValueError(f"Unsupported HTTP provider: {provider}")
        
        selected_model = model if model else default_model
        
        if provider == "deepseek":
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Translate: {text}"}
                        ],
                        "max_tokens": max_tokens
                    }
                )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Translate: {text}"}
                        ],
                        "max_tokens": max_tokens
                    }
                )
        
        if response.status_code != 200:
            if provider == "deepseek":
                error_detail = f"DeepSeek API error: {response.status_code}"
                try:
                    error_body = response.json()
                    if "error" in error_body:
                        error_detail += f" - {error_body['error'].get('message', 'Unknown error')}"
                except Exception:
                    error_detail += f" - Response: {response.text[:200]}"
                raise Exception(error_detail)
            else:
                raise Exception(f"{provider.capitalize()} API error")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    async def translate(self, text: str, llm_provider: str, api_token: str, 
                       model: Optional[str] = None) -> Tuple[str, List[Tuple[str, str, int, int]]]:
        """Main translation method that handles all providers."""
        is_english = self.detect_language(text)
        matches = self.dictionary_matcher.find_matches(text, is_english)
        
        translated = await self._call_llm_api(text, api_token, is_english, matches, llm_provider, model)
        
        return translated, matches
    
    def get_config(self) -> Dict:
        """Get default configuration for API tokens and models."""
        return {
            "default_tokens": {
                "chatgpt": os.getenv("OPENAI_API_KEY", ""),
                "chatgpt(azure)": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
                "qwen": os.getenv("DASHSCOPE_API_KEY", "")
            },
            "default_models": {
                "chatgpt": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                "chatgpt(azure)": os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                "deepseek": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
                "qwen": os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo")
            }
        }