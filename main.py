import logging
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from translator import TranslationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DICTIONARY_FILE = "QS-TB.csv"
STATIC_DIR = "static"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8099


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("Starting MRCT BOOK Translator service...")
    
    # Verify required files exist
    static_path = Path(STATIC_DIR)
    dict_path = Path(DICTIONARY_FILE)
    
    if not static_path.exists():
        logger.warning(f"Static directory not found: {static_path}")
    
    if not dict_path.exists():
        logger.warning(f"Dictionary file not found: {dict_path}")
    
    logger.info("Service startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MRCT BOOK Translator service...")


# Pydantic Models
class TranslationRequest(BaseModel):
    """Request model for translation endpoint."""
    text: str
    llm_provider: str
    api_token: str
    model: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, world!",
                "llm_provider": "chatgpt",
                "api_token": "your-api-token",
                "model": "gpt-3.5-turbo"
            }
        }


class TranslationResponse(BaseModel):
    """Response model for translation endpoint."""
    translated_text: str
    dictionary_matches: List[Tuple[str, str, int, int]]
    
    class Config:
        schema_extra = {
            "example": {
                "translated_text": "你好，世界！",
                "dictionary_matches": [["Hello", "你好", 0, 5]]
            }
        }


class ConfigResponse(BaseModel):
    """Response model for configuration endpoint."""
    supported_providers: List[str]
    default_provider: str
    dictionary_file: str
    dictionary_loaded: bool


# FastAPI App Configuration
app = FastAPI(
    title="MRCT BOOK Translator",
    description="English ⇄ Chinese translation service with technical dictionary support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8099", "http://127.0.0.1:8099"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize translation service with error handling
def create_translation_service() -> Optional[TranslationService]:
    """Create and initialize the translation service."""
    try:
        service = TranslationService(DICTIONARY_FILE)
        logger.info("Translation service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize translation service: {e}")
        return None


translation_service = create_translation_service()


# API Endpoints
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text using the specified LLM provider.
    
    Returns the translated text along with any dictionary matches found.
    """
    if translation_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service is not available"
        )
    
    try:
        translated, matches = await translation_service.translate(
            request.text, 
            request.llm_provider, 
            request.api_token, 
            request.model
        )
        
        return TranslationResponse(
            translated_text=translated,
            dictionary_matches=matches
        )
    
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get the current service configuration and status.
    
    Returns information about supported providers and dictionary status.
    """
    if translation_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service is not available"
        )
    
    try:
        return translation_service.get_config()
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@app.get("/download-dictionary")
async def download_dictionary():
    """
    Download the technical dictionary file.
    
    Returns the CSV file containing the dictionary data.
    """
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
    """
    Serve the main application interface.
    
    Returns the HTML interface for the translation service.
    """
    index_path = Path(STATIC_DIR) / "index.html"
    
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface not found")
    
    return FileResponse(str(index_path))


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check the health status of the service.
    
    Returns the current service status and component availability.
    """
    status = {
        "status": "healthy" if translation_service is not None else "degraded",
        "translation_service": translation_service is not None,
        "dictionary_file": Path(DICTIONARY_FILE).exists(),
        "static_files": Path(STATIC_DIR).exists()
    }
    
    return status


# Mount static files
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