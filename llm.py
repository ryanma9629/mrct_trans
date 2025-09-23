import os
import logging
from enum import Enum
from typing import Optional, Union

from openai import AsyncOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SupportedProvider(Enum):
    CHATGPT = "chatgpt"
    CHATGPT_AZURE = "chatgpt(azure)"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"

class APIError(Exception):
    pass

class LLMService:
    """Handles all interactions with different LLM providers."""

    def _get_llm_client(self, provider: SupportedProvider, api_token: str) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """Creates and configures the appropriate API client based on the provider."""
        if provider == SupportedProvider.CHATGPT_AZURE:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise APIError("AZURE_OPENAI_ENDPOINT not configured")
            return AsyncAzureOpenAI(
                api_key=api_token,
                azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            )
        
        # For all other OpenAI-compatible APIs
        base_url = None
        if provider == SupportedProvider.DEEPSEEK:
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1")
        elif provider == SupportedProvider.QWEN:
            base_url = os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        return AsyncOpenAI(
            api_key=api_token,
            base_url=base_url
        )

    def _get_model_for_provider(self, provider: SupportedProvider, model: Optional[str]) -> str:
        """Gets the appropriate model name, falling back to the default for the provider."""
        if model:
            return model
        
        env_var_map = {
            SupportedProvider.CHATGPT: "OPENAI_DEFAULT_MODEL",
            SupportedProvider.CHATGPT_AZURE: "AZURE_OPENAI_DEFAULT_MODEL",
            SupportedProvider.DEEPSEEK: "DEEPSEEK_DEFAULT_MODEL",
            SupportedProvider.QWEN: "QWEN_DEFAULT_MODEL"
        }
        default_model_map = {
            SupportedProvider.CHATGPT: "gpt-4o-mini",
            SupportedProvider.CHATGPT_AZURE: "gpt-4o-mini",
            SupportedProvider.DEEPSEEK: "deepseek-chat",
            SupportedProvider.QWEN: "qwen-turbo"
        }
        
        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var, default_model_map.get(provider, ''))
        return default_model_map.get(provider, '')

    def _validate_provider(self, provider: str) -> SupportedProvider:
        try:
            return SupportedProvider(provider)
        except ValueError:
            supported_providers = [p.value for p in SupportedProvider]
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: {supported_providers}")

    async def call_llm_api(self, text: str, system_prompt: str, provider: str, 
                           api_token: str, model: Optional[str] = None, temperature: float = 0.1) -> str:
        """
        Calls the specified LLM provider with the given text and system prompt.
        """
        provider_enum = self._validate_provider(provider)
        client = self._get_llm_client(provider_enum, api_token)
        selected_model = self._get_model_for_provider(provider_enum, model)

        try:
            response = await client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                temperature=temperature
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise APIError(f"Empty response from {provider_enum.value} API")
                
            return response.choices[0].message.content.strip()

        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider_enum.value} API error: {e}")
