import os
import csv
import re
import logging
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportedProvider(Enum):
    CHATGPT = "chatgpt"
    CHATGPT_AZURE = "chatgpt(azure)"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


@dataclass
class ProviderConfig:
    endpoint: str
    default_model: str
    max_tokens: int
    timeout: Optional[float] = None


class TranslationError(Exception):
    pass


class APIError(TranslationError):
    pass


class DictionaryMatcher:
    
    def __init__(self, csv_file: Union[str, Path]):
        self.en_to_cn: Dict[str, str] = {}
        self.cn_to_en: Dict[str, str] = {}
        self._load_dictionary(csv_file)
    
    def _load_dictionary(self, csv_file: Union[str, Path]) -> None:
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)
                
                for row_num, row in enumerate(reader, start=2):
                    if len(row) >= 2 and row[0].strip() and row[1].strip():
                        en_term = row[0].strip().lower()
                        cn_term = row[1].strip()
                        
                        if en_term not in self.en_to_cn:
                            self.en_to_cn[en_term] = cn_term
                        
                        if cn_term not in self.cn_to_en:
                            self.cn_to_en[cn_term] = row[0].strip()
                    elif len(row) >= 2:
                        logger.debug(f"Skipping empty entry at row {row_num}: {row}")
                        
            logger.info(f"Loaded {len(self.en_to_cn)} dictionary entries from {csv_path}")
            
        except (IOError, csv.Error) as e:
            raise TranslationError(f"Failed to load dictionary from {csv_path}: {e}")
    
    def find_matches(self, text: str, is_english: bool) -> List[Tuple[str, str, int, int]]:
        if not text.strip():
            return []
            
        matches = []
        dictionary = self.en_to_cn if is_english else self.cn_to_en
        
        if not dictionary:
            logger.warning("Dictionary is empty")
            return []
        
        sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)
        
        text_lower = text.lower() if is_english else text
        used_positions = set()
        
        for term in sorted_terms:
            try:
                if is_english:
                    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                    search_text = text_lower
                else:
                    pattern = re.compile(re.escape(term))
                    search_text = text
                
                for match in pattern.finditer(search_text):
                    start, end = match.span()
                    if not any(pos in used_positions for pos in range(start, end)):
                        matches.append((term, dictionary[term], start, end))
                        used_positions.update(range(start, end))
                        
            except re.error as e:
                logger.warning(f"Regex error for term '{term}': {e}")
                continue
        
        return sorted(matches, key=lambda x: x[2])


class TranslationService:
    
    TECHNICAL_TRANSLATION_TEMPERATURE = 0.1
    
    _PROVIDER_CONFIGS = {
        SupportedProvider.DEEPSEEK: ProviderConfig(
            endpoint=os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/chat/completions"),
            default_model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            max_tokens=4000,
            timeout=60.0
        ),
        SupportedProvider.QWEN: ProviderConfig(
            endpoint=os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            default_model=os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo"),
            max_tokens=2000,
            timeout=None
        )
    }
    
    def __init__(self, dictionary_file: Union[str, Path] = "QS-TB.csv"):
        try:
            self.dictionary_matcher = DictionaryMatcher(dictionary_file)
        except (FileNotFoundError, TranslationError) as e:
            logger.error(f"Failed to initialize dictionary matcher: {e}")
            raise
    
    def detect_language(self, text: str) -> bool:
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        non_space_chars = text.replace(' ', '')
        
        if not non_space_chars:
            raise ValueError("Text contains only spaces")
            
        chinese_ratio = len(chinese_chars) / len(non_space_chars)
        return chinese_ratio < 0.1
    
    def _prepare_dictionary_context(self, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        if not dictionary_matches:
            return ""
        
        dict_context = "\n\nTECHNICAL TERMINOLOGY: The following specialized terms must be translated exactly as specified:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nThese are established technical translations that must be used consistently. Translate all other content while preserving the exact terminology above."
        
        return dict_context
    
    def _prepare_system_prompt(self, is_english: bool, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        direction = "from English to Chinese" if is_english else "from Chinese to English"
        dict_context = self._prepare_dictionary_context(dictionary_matches)
        
        base_prompt = (
            f"You are a professional technical and scientific document translator specializing in {direction} translation. "
            f"Your task is to provide accurate, precise, and consistent translations that maintain the technical rigor of the original text.\n\n"
            f"Guidelines:\n"
            f"- Preserve all technical terminology, scientific concepts, and numerical data exactly\n"
            f"- Maintain the formal academic tone and structure of the original document\n"
            f"- Keep abbreviations, formulas, citations, and references unchanged\n"
            f"- Ensure consistency in terminology throughout the translation\n"
            f"- Do not add explanations, interpretations, or additional content\n"
            f"- Provide only the direct, professional translation of the input text"
        )
        
        return base_prompt + dict_context
    
    def _validate_provider(self, provider: str) -> SupportedProvider:
        try:
            return SupportedProvider(provider)
        except ValueError:
            supported_providers = [p.value for p in SupportedProvider]
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: {supported_providers}")
    
    async def _call_llm_api(self, text: str, api_token: str, is_english: bool, 
                           dictionary_matches: Optional[List[Tuple[str, str, int, int]]], 
                           provider: str, model: Optional[str] = None) -> str:
        provider_enum = self._validate_provider(provider)
        system_prompt = self._prepare_system_prompt(is_english, dictionary_matches)
        
        if provider_enum in [SupportedProvider.CHATGPT, SupportedProvider.CHATGPT_AZURE]:
            is_azure = provider_enum == SupportedProvider.CHATGPT_AZURE
            return await self._call_openai_api(text, api_token, system_prompt, is_azure, model)
        else:
            return await self._call_http_api(text, api_token, system_prompt, provider_enum, model)
    
    async def _call_openai_api(self, text: str, api_token: str, system_prompt: str, 
                              is_azure: bool, model: Optional[str]) -> str:
        try:
            if is_azure:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if not azure_endpoint:
                    raise APIError("AZURE_OPENAI_ENDPOINT not configured")
                
                client = AsyncAzureOpenAI(
                    api_key=api_token,
                    azure_endpoint=azure_endpoint,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
                )
                default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
                selected_model = model if model else default_model
                
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate: {text}"}
                    ],
                    max_tokens=2000,
                    temperature=self.TECHNICAL_TRANSLATION_TEMPERATURE
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
                    max_tokens=2000,
                    temperature=self.TECHNICAL_TRANSLATION_TEMPERATURE
                )
            
            if not response.choices or not response.choices[0].message.content:
                raise APIError("Empty response from OpenAI API")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            provider_name = "Azure OpenAI" if is_azure else "OpenAI"
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider_name} API error: {e}")
    
    async def _call_http_api(self, text: str, api_token: str, system_prompt: str, 
                            provider: SupportedProvider, model: Optional[str]) -> str:
        if provider not in self._PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported HTTP provider: {provider}")
        
        config = self._PROVIDER_CONFIGS[provider]
        selected_model = model if model else config.default_model
        
        request_data = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"}
            ],
            "max_tokens": config.max_tokens,
            "temperature": self.TECHNICAL_TRANSLATION_TEMPERATURE
        }
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            if config.timeout:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(config.endpoint, headers=headers, json=request_data)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(config.endpoint, headers=headers, json=request_data)
            
            if response.status_code != 200:
                await self._handle_http_error(response, provider)
            
            result = response.json()
            
            if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
                raise APIError(f"Empty response from {provider.value} API")
                
            return result["choices"][0]["message"]["content"].strip()
            
        except httpx.RequestError as e:
            raise APIError(f"Network error calling {provider.value} API: {e}")
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider.value} API error: {e}")
    
    async def _handle_http_error(self, response: httpx.Response, provider: SupportedProvider) -> None:
        error_detail = f"{provider.value} API error: {response.status_code}"
        
        try:
            error_body = response.json()
            if "error" in error_body:
                error_message = error_body["error"].get("message", "Unknown error")
                error_detail += f" - {error_message}"
        except Exception:
            error_detail += f" - Response: {response.text[:200]}"
        
        raise APIError(error_detail)
    
    async def translate(self, text: str, llm_provider: str, api_token: str, 
                       model: Optional[str] = None) -> Tuple[str, List[Tuple[str, str, int, int]]]:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not api_token.strip():
            raise ValueError("API token cannot be empty")
        
        try:
            is_english = self.detect_language(text)
            matches = self.dictionary_matcher.find_matches(text, is_english)
            
            logger.info(f"Translating text ({len(text)} chars) using {llm_provider}, "
                       f"found {len(matches)} dictionary matches in the input text")
            
            translated = await self._call_llm_api(
                text, api_token, is_english, matches, llm_provider, model
            )
            
            if not translated.strip():
                raise TranslationError("Translation result is empty")
            
            logger.info(f"Translation completed successfully ({len(translated)} chars)")
            return translated, matches
            
        except (ValueError, APIError):
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}")
    
    def get_config(self) -> Dict[str, Union[List[str], Dict[str, str]]]:
        return {
            "supported_providers": [provider.value for provider in SupportedProvider],
            "default_models": {
                SupportedProvider.CHATGPT.value: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                SupportedProvider.CHATGPT_AZURE.value: os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                SupportedProvider.DEEPSEEK.value: os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
                SupportedProvider.QWEN.value: os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo")
            },
            "default_tokens": {
                SupportedProvider.CHATGPT.value: os.getenv("OPENAI_API_KEY", ""),
                SupportedProvider.CHATGPT_AZURE.value: os.getenv("AZURE_OPENAI_API_KEY", ""),
                SupportedProvider.DEEPSEEK.value: os.getenv("DEEPSEEK_API_KEY", ""),
                SupportedProvider.QWEN.value: os.getenv("DASHSCOPE_API_KEY", "")
            }
        }