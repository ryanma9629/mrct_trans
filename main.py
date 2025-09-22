import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from contextlib import asynccontextmanager

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
    rag: bool = False
    chapter_number: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, world!",
                "llm_provider": "chatgpt",
                "api_token": "your-api-token",
                "model": "gpt-3.5-turbo",
                "rag": False,
                "chapter_number": 1
            }
        }


class TranslationResponse(BaseModel):
    translated_text: str
    dictionary_matches: List[Tuple[str, str, int, int]]
    retrieved_contexts: Optional[List[str]] = []
    
    class Config:
        schema_extra = {
            "example": {
                "translated_text": "你好，世界！",
                "dictionary_matches": [["Hello", "你好", 0, 5]],
                "retrieved_contexts": ["Context from the book..."]
            }
        }


class ConfigResponse(BaseModel):
    supported_providers: List[str]
    default_models: Dict[str, str]
    default_tokens: Dict[str, str]
    default_provider: str
    dictionary_file: str
    dictionary_loaded: bool


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


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    if translation_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service is not available"
        )
    
    try:
        translated, matches, contexts = await translation_service.translate(
            request.text, 
            request.llm_provider, 
            request.api_token, 
            request.model,
            request.rag,
            request.chapter_number
        )
        
        return TranslationResponse(
            translated_text=translated,
            dictionary_matches=matches,
            retrieved_contexts=contexts
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


@app.get("/download-dictionary")
async def download_dictionary():
    dict_path = Path(DICTIONARY_FILE)
    
    if not dict_path.exists():
        raise HTTPException(status_code=404, detail="Dictionary file not found")
    
    return FileResponse(
        path=str(dict_path),
        filename=DICTIONARY_FILE,
        media_type="text/csv"
    )


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