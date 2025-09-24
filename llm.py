import logging
import os
from enum import Enum
from typing import Optional, Union

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Load environment configuration
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

    def _get_llm_client(
        self, provider: SupportedProvider, api_token: str
    ) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """Create and configure the appropriate API client for the provider."""
        if provider == SupportedProvider.CHATGPT_AZURE:
            return self._create_azure_client(api_token)
        else:
            return self._create_openai_compatible_client(provider, api_token)

    def _create_azure_client(self, api_token: str) -> AsyncAzureOpenAI:
        """Create Azure OpenAI client with configuration validation."""
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise APIError("AZURE_OPENAI_ENDPOINT not configured")

        return AsyncAzureOpenAI(
            api_key=api_token,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        )

    def _create_openai_compatible_client(self, provider: SupportedProvider, api_token: str) -> AsyncOpenAI:
        """Create OpenAI-compatible client with provider-specific base URL."""
        base_url = self._get_provider_base_url(provider)
        return AsyncOpenAI(api_key=api_token, base_url=base_url)

    def _get_provider_base_url(self, provider: SupportedProvider) -> Optional[str]:
        """Get the base URL for the given provider."""
        url_mapping = {
            SupportedProvider.DEEPSEEK: os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/"),
            SupportedProvider.QWEN: os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        }
        return url_mapping.get(provider)

    def _get_model_for_provider(
        self, provider: SupportedProvider, model: Optional[str]
    ) -> str:
        """Get user-selected model or fallback to provider default."""
        if model and model.strip():
            logger.info(f"Using user-selected model: {model} for {provider.value}")
            return model.strip()

        default_model = self._get_default_model(provider)
        logger.info(f"Using default model: {default_model} for {provider.value}")
        return default_model

    def _get_default_model(self, provider: SupportedProvider) -> str:
        """Get the default model for a provider."""
        default_models = {
            SupportedProvider.CHATGPT: "gpt-4o-mini",
            SupportedProvider.CHATGPT_AZURE: "gpt-4o-mini",
            SupportedProvider.DEEPSEEK: "deepseek-chat",
            SupportedProvider.QWEN: "qwen-turbo",
        }
        return default_models.get(provider, "gpt-4o-mini")

    def _validate_provider(self, provider: str) -> SupportedProvider:
        """Validate and convert provider string to enum."""
        try:
            return SupportedProvider(provider)
        except ValueError:
            supported = [p.value for p in SupportedProvider]
            raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

    async def call_llm_api(
        self,
        text: str,
        system_prompt: str,
        provider: str,
        api_token: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Call the specified LLM provider API for translation."""
        provider_enum = self._validate_provider(provider)
        client = self._get_llm_client(provider_enum, api_token)
        selected_model = self._get_model_for_provider(provider_enum, model)

        try:
            response = await self._make_api_call(
                client, selected_model, system_prompt, text, temperature
            )
            return self._extract_response_content(response, provider_enum)

        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider_enum.value} API error: {e}")

    async def _make_api_call(self, client, model: str, system_prompt: str, text: str, temperature: float):
        """Make the actual API call to the LLM provider."""
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"},
            ],
            temperature=temperature,
        )

    def _extract_response_content(self, response, provider_enum: SupportedProvider) -> str:
        """Extract and validate the response content."""
        if not response.choices or not response.choices[0].message.content:
            raise APIError(f"Empty response from {provider_enum.value} API")

        return response.choices[0].message.content.strip()
