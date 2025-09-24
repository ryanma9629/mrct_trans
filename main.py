import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from contextlib import asynccontextmanager
from datetime import datetime
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from translator import TranslationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DICTIONARY_FILE = "QS-TB.csv"
STATIC_DIR = "static"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8099


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MRCT BOOK Translator service...")
    
    static_path = Path(STATIC_DIR)
    dict_path = Path(DICTIONARY_FILE)
    
    if not static_path.exists():
        logger.warning(f"Static directory not found: {static_path}")
    
    if not dict_path.exists():
        logger.warning(f"Dictionary file not found: {dict_path}")
    
    logger.info("Service startup complete")
    
    yield
    
    logger.info("Shutting down MRCT BOOK Translator service...")


class TranslationRequest(BaseModel):
    text: str
    llm_provider: str
    api_token: str
    model: Optional[str] = None
    use_context: bool = False
    chapter_number: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, world!",
                "llm_provider": "chatgpt",
                "api_token": "your-api-token",
                "model": "gpt-3.5-turbo",
                "use_context": False,
                "chapter_number": 1
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
    default_tokens: Dict[str, str]
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
    try:
        service = TranslationService(DICTIONARY_FILE)
        logger.info("Translation service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize translation service: {e}")
        return None


translation_service = create_translation_service()

# Translation history storage (in-memory for simplicity, stores last 5 translations)
translation_history: deque = deque(maxlen=5)


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    if translation_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service is not available"
        )
    
    try:
        translated, matches, context = await translation_service.translate(
            request.text,
            request.llm_provider,
            request.api_token,
            request.model,
            request.use_context,
            request.chapter_number
        )

        # Detect source and target languages
        is_english = translation_service.detect_language(request.text)
        source_lang = "en" if is_english else "zh"
        target_lang = "zh" if is_english else "en"

        # Store in translation history
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

        return TranslationResponse(
            translated_text=translated,
            dictionary_matches=matches,
            context=context
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    if translation_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service is not available"
        )
    
    try:
        # Get the basic config from the translation service
        basic_config = translation_service.get_config()
        
        # Extract the configuration data with proper type casting
        supported_providers = basic_config.get("supported_providers", [])
        default_models = basic_config.get("default_models", {})
        default_tokens = basic_config.get("default_tokens", {})
        
        # Ensure we have the right types
        if not isinstance(supported_providers, list):
            supported_providers = []
        if not isinstance(default_models, dict):
            default_models = {}
        if not isinstance(default_tokens, dict):
            default_tokens = {}
        
        # Add the missing fields required by ConfigResponse
        return ConfigResponse(
            supported_providers=supported_providers,
            default_models=default_models,
            default_tokens=default_tokens,
            default_provider=supported_providers[0] if supported_providers else "chatgpt",
            dictionary_file=DICTIONARY_FILE,
            dictionary_loaded=Path(DICTIONARY_FILE).exists()
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@app.get("/history", response_model=List[TranslationHistoryItem])
async def get_translation_history():
    """Get the recent translation history (up to 5 translations)."""
    try:
        # Return history items in reverse chronological order (most recent first)
        return list(reversed(translation_history))
    except Exception as e:
        logger.error(f"Failed to get translation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.delete("/history")
async def clear_translation_history():
    """Clear the translation history."""
    try:
        translation_history.clear()
        logger.info("Translation history cleared")
        return {"message": "Translation history cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear translation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@app.get("/chapters")
async def get_chapters():
    """Scans the context_data directory and returns a sorted list of chapter numbers."""
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
        logger.info(f"Found available chapters: {sorted_chapters}")
        return sorted_chapters
    except Exception as e:
        logger.error(f"Failed to scan for chapters: {e}")
        return []


@app.get("/")
async def serve_index():
    index_path = Path(STATIC_DIR) / "index.html"
    
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface not found")
    
    return FileResponse(str(index_path))


@app.get("/health")
async def health_check():
    status = {
        "status": "healthy" if translation_service is not None else "degraded",
        "translation_service": translation_service is not None,
        "dictionary_file": Path(DICTIONARY_FILE).exists(),
        "static_files": Path(STATIC_DIR).exists()
    }
    
    return status


if Path(STATIC_DIR).exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Static files mounted from {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")


# Application entry point
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {DEFAULT_HOST}:{DEFAULT_PORT}")
    
    uvicorn.run(
        "main:app", 
        host=DEFAULT_HOST, 
        port=DEFAULT_PORT, 
        reload=False,
        log_level="info"
    )