from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from translator import TranslationService

app = FastAPI()
translation_service = TranslationService()

class TranslationRequest(BaseModel):
    text: str
    llm_provider: str
    api_token: str
    model: Optional[str] = None

class TranslationResponse(BaseModel):
    translated_text: str
    dictionary_matches: List[Tuple[str, str, int, int]]


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    return translation_service.get_config()


@app.get("/download-dictionary")
async def download_dictionary():
    return FileResponse(
        path="QS-TB.csv",
        filename="QS-TB.csv",
        media_type="text/csv"
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="localhost", port=8099, reload=False)