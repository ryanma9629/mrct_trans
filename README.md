# MRCT BOOK Translator

**A powerful web-based English ⇄ Chinese translation tool with technical dictionary support and RAG-enhanced context-aware translation**

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![License](https://img.shields.io/badge/License-Educational-orange)
![Platform](https://img.shields.io/badge/Platform-Cross--Platform-lightgrey)

## Features

- **Bidirectional Translation**: Seamlessly translate between English and Chinese with automatic language detection
- **Multiple AI Providers**: Support for Qwen, DeepSeek, ChatGPT, and Azure OpenAI
- **Technical Dictionary**: Built-in [`QS-TB.csv`](QS-TB.csv) dictionary with 2,700+ technical terms
- **RAG-Enhanced Translation**: Context-aware translation using book chapters and PDF content
- **Smart Context Retrieval**: Automatic similarity-based context matching for improved accuracy
- **PDF Processing**: Extract and process context from PDF files using [`load_pdfs.py`](load_pdfs.py)
- **Dictionary Highlighting**: Visual highlighting of translated technical terms
- **Modern Web Interface**: Clean, responsive design with real-time context feedback
- **Real-time Translation**: Fast API responses with loading indicators and similarity scores
- **Copy to Clipboard**: One-click copying of translation results
- **Privacy-First**: All processing runs locally on your machine

## Installation

### Windows Installation
```cmd
# 1. Clone the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create configuration file
copy .env.example .env
```

### macOS & Linux Installation
```bash
# 1. Clone the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create configuration file
cp .env.example .env
```

## Configuration

### 1. API Keys Setup
Edit the `.env` file and add your API keys. See [`API_KEY_Application_Guide.md`](API_KEY_Application_Guide.md) for details.
```bash
# Qwen (Recommended for Chinese translations)
DASHSCOPE_API_KEY=sk-your_dashscope_key_here

# DeepSeek (Cost-effective option)
DEEPSEEK_API_KEY=sk-your_deepseek_key_here

# OpenAI ChatGPT (Optional)
OPENAI_API_KEY=your_openai_api_key_here

# Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### 2. RAG Context Setup (Optional)
For enhanced context-aware translation:
1.  Create a `pdf/` directory and place your PDF files inside.
2.  Run the processing script: `python load_pdfs.py`

This will create context data in the `context_data/` directory, enabling RAG-enhanced translation.

## Usage

### Starting the Application
```bash
# Start the server (use python3 on macOS/Linux if needed)
python main.py
```
The application will be available at `http://localhost:8099`.

### Using the Web Interface
1.  **Select LLM Provider**: Choose your preferred AI provider and model.
2.  **Enter API Token**: Provide your API key.
3.  **Configure RAG** (Optional): Enable "Retrieve context" and select a chapter.
4.  **Translate**: Enter text and click "Translate" or press `Ctrl+Enter`.

## API Reference

### Translation Endpoint
`POST /translate`
```json
{
  "text": "Hello world",
  "llm_provider": "qwen",
  "api_token": "your-token",
  "model": "qwen-turbo",
  "use_context": true,
  "chapter_number": 1
}
```

### Other Endpoints
-   `GET /config`: Get application configuration.
-   `GET /chapters`: Get a list of available context chapters.
-   `GET /health`: Check the health status of the service.

## Project Structure
```
mrct_trans/
├── main.py                      # FastAPI web server and API endpoints
├── translator.py                # Translation service core logic with RAG
├── llm.py                       # LLM provider integrations
├── load_pdfs.py                 # PDF processing for RAG context
├── static/                      # Web frontend files (HTML, CSS, JS)
├── pdf/                         # Source PDF files for context
├── context_data/                # Processed text files (auto-generated)
├── QS-TB.csv                    # Technical dictionary
├── .env                         # Local configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
