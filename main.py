"""MRCT BOOK Translator - Main Application

A FastAPI-based translation service for English ⇄ Chinese translation
with technical dictionary support and context-aware enhancement.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from contextlib import asynccontextmanager
from datetime import datetime
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import json

from translator import TranslationService
from cache import cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application constants
DICTIONARY_FILE = "QS-TB.csv"
STATIC_DIR = "static"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8099
HISTORY_SIZE = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting MRCT BOOK Translator service")

    static_path = Path(STATIC_DIR)
    dict_path = Path(DICTIONARY_FILE)

    # Validate critical paths
    if not static_path.exists():
        logger.warning(f"Static directory missing: {static_path}")

    if not dict_path.exists():
        logger.warning(f"Dictionary file missing: {dict_path}")

    logger.info("Service startup completed successfully")

    yield

    # Shutdown
    logger.info("Shutting down MRCT BOOK Translator service")


class TranslationRequest(BaseModel):
    text: str
    llm_provider: str
    api_token: str
    model: Optional[str] = None
    use_context: bool = False
    chapter_number: Optional[int] = None
    stream: bool = False

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, world!",
                "llm_provider": "chatgpt",
                "api_token": "your-api-token",
                "model": "gpt-3.5-turbo",
                "use_context": False,
                "chapter_number": 1,
                "stream": False
            }
        }


class TranslationResponse(BaseModel):
    translated_text: str
    dictionary_matches: List[Tuple[str, str, int, int]]
    context: Optional[Dict[str, Union[str, float]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "translated_text": "你好，世界！",
                "dictionary_matches": [["Hello", "你好", 0, 5]],
                "context": {"text": "Context from the book...", "score": 0.95}
            }
        }


class ConfigResponse(BaseModel):
    supported_providers: List[str]
    default_models: Dict[str, str]
    tokens_available: Dict[str, bool]
    default_provider: str
    dictionary_file: str
    dictionary_loaded: bool


class TranslationHistoryItem(BaseModel):
    timestamp: datetime
    original_text: str
    translated_text: str
    source_language: str  # "en" or "zh"
    target_language: str  # "zh" or "en"
    llm_provider: str
    model: Optional[str] = None
    use_context: bool = False
    chapter_number: Optional[int] = None
    dictionary_matches: List[Tuple[str, str, int, int]]
    context: Optional[Dict[str, Union[str, float]]] = None

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2023-12-01T12:00:00",
                "original_text": "Hello world",
                "translated_text": "你好世界",
                "source_language": "en",
                "target_language": "zh",
                "llm_provider": "qwen",
                "model": "qwen-turbo",
                "use_context": False,
                "chapter_number": None,
                "dictionary_matches": [],
                "context": None
            }
        }


app = FastAPI(
    title="MRCT BOOK Translator",
    description="English ⇄ Chinese translation service with technical dictionary support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8099", "http://127.0.0.1:8099"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def create_translation_service() -> Optional[TranslationService]:
    """Initialize the translation service with comprehensive error handling.

    Returns:
        TranslationService instance if successful, None otherwise
    """
    try:
        service = TranslationService(DICTIONARY_FILE)
        logger.info("Translation service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Translation service initialization failed: {e}")
        return None


# Global translation service instance
translation_service = create_translation_service()

# Translation history storage (in-memory, stores last N translations)
translation_history: deque = deque(maxlen=HISTORY_SIZE)


def _store_translation_history(
    request: TranslationRequest,
    translated: str,
    matches: List[Tuple[str, str, int, int]],
    context: Optional[Dict[str, Union[str, float]]]
) -> None:
    """Store completed translation in history with automatic language detection.

    Args:
        request: Original translation request
        translated: Final translated text
        matches: Dictionary matches found in the text
        context: Context information if used
    """
    is_english = translation_service.detect_language(request.text)
    source_lang = "en" if is_english else "zh"
    target_lang = "zh" if is_english else "en"

    history_item = TranslationHistoryItem(
        timestamp=datetime.now(),
        original_text=request.text,
        translated_text=translated,
        source_language=source_lang,
        target_language=target_lang,
        llm_provider=request.llm_provider,
        model=request.model,
        use_context=request.use_context,
        chapter_number=request.chapter_number,
        dictionary_matches=matches,
        context=context
    )
    translation_history.append(history_item)
    logger.debug(f"Translation stored in history - {source_lang} to {target_lang}, {len(translated)} chars")


def _validate_translation_service() -> None:
    """Validate that translation service is available and raise HTTP error if not.

    Raises:
        HTTPException: 503 if translation service is unavailable
    """
    if translation_service is None:
        logger.error("Translation service validation failed - service unavailable")
        raise HTTPException(
            status_code=503,
            detail="Translation service is not available"
        )


def _build_config_response(basic_config: Dict) -> ConfigResponse:
    """Build configuration response with safe type casting and validation.

    Args:
        basic_config: Raw configuration data from translation service

    Returns:
        ConfigResponse with validated and properly typed configuration
    """
    # Extract and validate configuration data
    supported_providers = basic_config.get("supported_providers", [])
    default_models = basic_config.get("default_models", {})
    tokens_available = basic_config.get("tokens_available", {})

    # Ensure correct types to prevent runtime errors
    if not isinstance(supported_providers, list):
        supported_providers = []
    if not isinstance(default_models, dict):
        default_models = {}
    if not isinstance(tokens_available, dict):
        tokens_available = {}

    return ConfigResponse(
        supported_providers=supported_providers,
        default_models=default_models,
        tokens_available=tokens_available,
        default_provider=supported_providers[0] if supported_providers else "chatgpt",
        dictionary_file=DICTIONARY_FILE,
        dictionary_loaded=Path(DICTIONARY_FILE).exists()
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Perform text translation with dictionary matching and optional context enhancement.

    Args:
        request: Translation request with text, provider, and settings

    Returns:
        TranslationResponse with translated text, matches, and context info

    Raises:
        HTTPException: 503 if service unavailable, 500 if translation fails
    """
    _validate_translation_service()

    try:
        logger.info(f"Translation request - {len(request.text)} chars, provider: {request.llm_provider}")

        # Perform translation
        translated, matches, context = await translation_service.translate(
            request.text,
            request.llm_provider,
            request.api_token,
            request.model,
            request.use_context,
            request.chapter_number
        )

        # Store in history and return response
        _store_translation_history(request, translated, matches, context)

        logger.info(f"Translation completed - {len(translated)} chars output")
        return TranslationResponse(
            translated_text=translated,
            dictionary_matches=matches,
            context=context
        )

    except Exception as e:
        logger.error(f"Translation request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate-stream")
async def translate_text_stream(request: TranslationRequest):
    """Stream translation results in real-time using Server-Sent Events.

    Args:
        request: Translation request with streaming enabled

    Returns:
        StreamingResponse with real-time translation chunks

    Raises:
        HTTPException: 400 if streaming not enabled, 503 if service unavailable
    """
    _validate_translation_service()

    if not request.stream:
        logger.warning("Streaming endpoint called without streaming enabled")
        raise HTTPException(status_code=400, detail="Streaming must be enabled for this endpoint")

    logger.info(f"Streaming translation request - {len(request.text)} chars, provider: {request.llm_provider}")

    async def generate_translation_stream():
        complete_translation = ""
        try:
            # Prepare translation components
            is_english = translation_service.detect_language(request.text)
            matches = translation_service.dictionary_matcher.find_matches(request.text, is_english)

            # Send initial metadata
            metadata = {
                "type": "metadata",
                "source_language": "en" if is_english else "zh",
                "target_language": "zh" if is_english else "en",
                "dictionary_matches": matches,
                "use_context": request.use_context
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # Handle context retrieval if needed
            context_info = None
            if request.use_context and request.chapter_number:
                if translation_service.context_enabled:
                    context_result = translation_service._retrieve_text_context(request.text, request.chapter_number)
                    if context_result:
                        context, similarity = context_result
                        context_info = {"text": context, "score": similarity}

                        context_data = {
                            "type": "context",
                            "context": context_info
                        }
                        yield f"data: {json.dumps(context_data)}\n\n"

            # Perform streaming translation and accumulate complete text
            async for chunk in translation_service.translate_stream(
                request.text,
                request.llm_provider,
                request.api_token,
                request.model,
                request.use_context,
                request.chapter_number
            ):
                complete_translation += chunk
                chunk_data = {
                    "type": "chunk",
                    "content": chunk
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Store complete translation in history
            _store_translation_history(request, complete_translation, matches, context_info)

            # Send completion signal
            completion_data = {
                "type": "complete",
                "dictionary_matches": matches,
                "context": context_info
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_translation_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get application configuration including supported providers and models.

    Returns:
        ConfigResponse with available providers, models, and token availability

    Raises:
        HTTPException: 503 if service unavailable, 500 if config retrieval fails
    """
    _validate_translation_service()

    try:
        logger.debug("Retrieving application configuration")
        # Get the basic config from the translation service
        basic_config = translation_service.get_config()

        # Get configuration with safe type casting
        config = _build_config_response(basic_config)
        logger.debug(f"Configuration retrieved - {len(config.supported_providers)} providers available")
        return config
    except Exception as e:
        logger.error(f"Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@app.get("/token/{provider}")
async def get_api_token(provider: str):
    """Retrieve API token for a specific provider from environment variables.

    Args:
        provider: LLM provider name (chatgpt, deepseek, etc.)

    Returns:
        Dict containing the API token if available

    Raises:
        HTTPException: 404 if provider not found or token not available
    """
    # Map provider to environment variable
    token_map = {
        "chatgpt": "OPENAI_API_KEY",
        "chatgpt(azure)": "AZURE_OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY"
    }

    if provider not in token_map:
        logger.warning(f"Token requested for unknown provider: {provider}")
        raise HTTPException(status_code=404, detail="Provider not found")

    env_var = token_map[provider]
    token = os.getenv(env_var, "").strip()

    if not token:
        logger.debug(f"No token available for provider: {provider}")
        raise HTTPException(status_code=404, detail="Token not available")

    logger.debug(f"Token retrieved successfully for provider: {provider}")
    return {"token": token}


@app.get("/history", response_model=List[TranslationHistoryItem])
async def get_translation_history():
    """Retrieve recent translation history in reverse chronological order.

    Returns:
        List of recent translation history items (up to 5 most recent)

    Raises:
        HTTPException: 500 if history retrieval fails
    """
    try:
        history_count = len(translation_history)
        logger.debug(f"Retrieving translation history - {history_count} items available")
        # Return history items in reverse chronological order (most recent first)
        return list(reversed(translation_history))
    except Exception as e:
        logger.error(f"Translation history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.delete("/history")
async def clear_translation_history():
    """Clear all translation history entries.

    Returns:
        Success message confirming history was cleared

    Raises:
        HTTPException: 500 if clearing fails
    """
    try:
        items_cleared = len(translation_history)
        translation_history.clear()
        logger.info(f"Translation history cleared - {items_cleared} items removed")
        return {"message": "Translation history cleared successfully"}
    except Exception as e:
        logger.error(f"Translation history clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@app.get("/chapters")
async def get_chapters():
    """Scan context_data directory and return available chapter numbers.

    Returns:
        Sorted list of available chapter numbers for context enhancement
    """
    context_path = Path("context_data")
    if not context_path.is_dir():
        logger.warning(f"Context data directory not found: {context_path}")
        return []

    try:
        chapter_numbers = set()
        for f in context_path.glob("Chapter*.txt"):
            match = re.search(r"Chapter(\d+)\.txt", f.name)
            if match:
                chapter_numbers.add(int(match.group(1)))

        sorted_chapters = sorted(list(chapter_numbers))
        logger.info(f"Found {len(sorted_chapters)} available chapters: {sorted_chapters}")
        return sorted_chapters
    except Exception as e:
        logger.error(f"Chapter scanning failed: {e}")
        return []


@app.get("/")
async def serve_index():
    """Serve the main web interface.

    Returns:
        Main HTML interface file

    Raises:
        HTTPException: 404 if interface file not found
    """
    index_path = Path(STATIC_DIR) / "index.html"

    if not index_path.exists():
        logger.error(f"Main interface file not found: {index_path}")
        raise HTTPException(status_code=404, detail="Interface not found")

    return FileResponse(str(index_path))


@app.get("/health")
async def health_check():
    """Perform application health check and return system status.

    Returns:
        Dict with health status and component availability
    """
    service_available = translation_service is not None
    dict_available = Path(DICTIONARY_FILE).exists()
    static_available = Path(STATIC_DIR).exists()

    status = "healthy" if service_available else "degraded"
    logger.debug(f"Health check - Status: {status}, Service: {service_available}")

    return {
        "status": status,
        "translation_service": service_available,
        "dictionary_file": dict_available,
        "static_files": static_available
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """Retrieve cache statistics and performance metrics.

    Returns:
        Dict with cache statistics and timestamp

    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    try:
        logger.debug("Retrieving cache statistics")
        stats = cache_manager.get_all_stats()
        return {
            "caches": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.post("/cache/clear")
async def clear_cache():
    """Clear all application caches.

    Returns:
        Success message confirming caches were cleared

    Raises:
        HTTPException: 500 if cache clearing fails
    """
    try:
        cache_manager.clear_all()
        logger.info("All caches cleared via API request")
        return {"message": "All caches cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")


@app.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired entries from all caches.

    Returns:
        Dict with cleanup results and number of entries removed

    Raises:
        HTTPException: 500 if cleanup fails
    """
    try:
        cleaned = cache_manager.cleanup_all()
        logger.info(f"Cache cleanup completed - {cleaned} entries removed")
        return {
            "message": "Cache cleanup completed",
            "cleaned_entries": cleaned
        }
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup caches: {str(e)}")


# Mount static files if directory exists
if Path(STATIC_DIR).exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Static files mounted successfully from {STATIC_DIR}")
else:
    logger.warning(f"Static directory missing: {STATIC_DIR} - web interface may not work")


# Application entry point
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting MRCT BOOK Translator server on {DEFAULT_HOST}:{DEFAULT_PORT}")

    uvicorn.run(
        "main:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
        log_level="info"
    )