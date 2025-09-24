"""LLM Service - Multi-provider language model integration.

Provides unified interface for multiple LLM providers:
- OpenAI GPT models (standard and Azure)
- Alibaba Qwen models via DashScope
- DeepSeek models

Features:
- Async streaming support for real-time translation
- Provider-specific configuration and error handling
- Automatic model selection with user override capability
- Comprehensive error handling and logging
"""

import asyncio
import logging
import os
from enum import Enum
from typing import Optional, Union

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Load environment configuration
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timeout configuration constants
DEFAULT_REQUEST_TIMEOUT = 30.0  # 30 seconds for regular requests
STREAMING_REQUEST_TIMEOUT = 60.0  # 60 seconds for streaming requests
CONNECT_TIMEOUT = 10.0  # 10 seconds to establish connection


class SupportedProvider(Enum):
    CHATGPT = "chatgpt"
    CHATGPT_AZURE = "chatgpt(azure)"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class APIError(Exception):
    pass


class LLMService:
    """Unified service for interacting with multiple LLM providers.

    Provides a consistent interface for translation requests across different
    language model providers with automatic configuration and error handling.
    """

    def _get_llm_client(
        self, provider: SupportedProvider, api_token: str
    ) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """Create and configure the appropriate API client for the provider.

        Args:
            provider: LLM provider enum
            api_token: API token for authentication

        Returns:
            Configured async client for the provider

        Raises:
            APIError: If client configuration fails
        """
        if provider == SupportedProvider.CHATGPT_AZURE:
            return self._create_azure_client(api_token)
        else:
            return self._create_openai_compatible_client(provider, api_token)

    def _create_azure_client(self, api_token: str) -> AsyncAzureOpenAI:
        """Create Azure OpenAI client with configuration validation.

        Args:
            api_token: Azure OpenAI API key

        Returns:
            Configured Azure OpenAI client

        Raises:
            APIError: If Azure endpoint is not configured
        """
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise APIError("AZURE_OPENAI_ENDPOINT not configured")

        return AsyncAzureOpenAI(
            api_key=api_token,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        )

    def _create_openai_compatible_client(self, provider: SupportedProvider, api_token: str) -> AsyncOpenAI:
        """Create OpenAI-compatible client with provider-specific base URL.

        Args:
            provider: LLM provider enum
            api_token: API token for the provider

        Returns:
            Configured OpenAI-compatible client

        Raises:
            APIError: If provider is not supported or configuration fails
        """
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
            logger.debug(f"Using user-selected model: {model} for {provider.value}")
            return model.strip()

        default_model = self._get_default_model(provider)
        logger.debug(f"Using default model: {default_model} for {provider.value}")
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

        except APIError:
            # Re-raise APIError as-is (already has user-friendly message)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {provider_enum.value} API call: {e}")
            raise APIError(f"{provider_enum.value} translation service error: {str(e)}")

    async def _make_api_call(self, client, model: str, system_prompt: str, text: str, temperature: float):
        """Make the actual API call to the LLM provider with timeout handling.

        Args:
            client: Configured LLM client
            model: Model name to use
            system_prompt: System prompt for translation
            text: Text to translate
            temperature: Temperature for generation

        Returns:
            API response object

        Raises:
            APIError: If request times out or fails
        """
        try:
            # Add timeout wrapper around the API call
            return await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate: {text}"},
                    ],
                    temperature=temperature,
                ),
                timeout=DEFAULT_REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM API request timed out after {DEFAULT_REQUEST_TIMEOUT}s for model {model}")
            raise APIError(f"Request timed out after {DEFAULT_REQUEST_TIMEOUT} seconds. Please check your network connection or try again later.")
        except Exception as e:
            logger.error(f"LLM API request failed: {str(e)}")
            # Re-raise with more user-friendly message for common issues
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise APIError("Network connection failed. Please check your internet connection and try again.")
            elif "unauthorized" in str(e).lower() or "403" in str(e) or "401" in str(e):
                raise APIError("API authentication failed. Please check your API token and try again.")
            else:
                raise APIError(f"Translation request failed: {str(e)}")

    def _extract_response_content(self, response, provider_enum: SupportedProvider) -> str:
        """Extract and validate the response content."""
        if not response.choices or not response.choices[0].message.content:
            raise APIError(f"Empty response from {provider_enum.value} API")

        return response.choices[0].message.content.strip()

    async def call_llm_api_stream(
        self,
        text: str,
        system_prompt: str,
        provider: str,
        api_token: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """Call the specified LLM provider API for streaming translation."""
        provider_enum = self._validate_provider(provider)
        client = self._get_llm_client(provider_enum, api_token)
        selected_model = self._get_model_for_provider(provider_enum, model)

        try:
            response = await self._make_streaming_api_call(
                client, selected_model, system_prompt, text, temperature
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APIError:
            # Re-raise APIError as-is (already has user-friendly message)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {provider_enum.value} streaming API call: {e}")
            raise APIError(f"{provider_enum.value} streaming translation service error: {str(e)}")

    async def _make_streaming_api_call(self, client, model: str, system_prompt: str, text: str, temperature: float):
        """Make a streaming API call to the LLM provider with error handling.

        Args:
            client: Configured LLM client
            model: Model name to use
            system_prompt: System prompt for translation
            text: Text to translate
            temperature: Temperature for generation

        Returns:
            Streaming API response

        Raises:
            APIError: If request setup fails
        """
        try:
            # Create the streaming request (no timeout wrapper for streaming)
            # The OpenAI client handles timeouts internally for streaming
            return await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate: {text}"},
                ],
                temperature=temperature,
                stream=True
            )
        except Exception as e:
            logger.error(f"Streaming LLM API request failed: {str(e)}")
            # Re-raise with more user-friendly message for common issues
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise APIError("Network connection failed. Please check your internet connection and try again.")
            elif "unauthorized" in str(e).lower() or "403" in str(e) or "401" in str(e):
                raise APIError("API authentication failed. Please check your API token and try again.")
            else:
                raise APIError(f"Streaming translation request failed: {str(e)}")
