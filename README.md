# MRCT BOOK Translator

🌐 **A powerful web-based English ⇄ Chinese translation tool with technical dictionary support**

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![License](https://img.shields.io/badge/License-Educational-orange)
![Platform](https://img.shields.io/badge/Platform-Cross--Platform-lightgrey)

## 🚀 Features

- **🔄 Bidirectional Translation**: Seamlessly translate between English and Chinese
- **🤖 Multiple AI Providers**: Support for Qwen, DeepSeek, ChatGPT, and Azure OpenAI
- **📚 Technical Dictionary**: Built-in QS-TB.csv dictionary with 2,700+ technical terms
- **🎯 Smart Language Detection**: Automatically detects input language
- **💡 Context-Aware Translation**: Uses dictionary terms for accurate technical translations
- **🖥️ Modern Web Interface**: Clean, responsive design for easy use
- **⚡ Real-time Translation**: Fast API responses with loading indicators
- **📋 Copy to Clipboard**: One-click copying of translation results
- **🔒 Privacy-First**: All processing runs locally on your machine

## 🛠️ Supported AI Providers

| Provider | Model | Strengths | Cost |
|----------|-------|-----------|------|
| **Alibaba Qwen** | qwen-turbo | Best for Chinese translations | Low |
| **DeepSeek** | deepseek-chat | Cost-effective, good quality | Very Low |
| **OpenAI ChatGPT** | gpt-4o-mini | Most reliable, high quality | Medium |
| **Azure OpenAI** | gpt-4o-mini | Enterprise-grade reliability | Medium |

## 📸 Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│                    MRCT BOOK Translator                    │
│              English ⇄ Chinese Translation                 │
├─────────────────┬───────────────────────────────────────────┤
│ LLM Provider:   │ Text to Translate:                      │
│ ▼ Qwen          │ ┌─────────────────────────────────────┐ │
│                 │ │ Enter text to translate...          │ │
│ API Token:      │ │                                     │ │
│ ********        │ │                                     │ │
│                 │ └─────────────────────────────────────┘ │
│ Technical Dict. │                                         │
│ 📥 Download CSV │ [Translate]                            │
│                 │                                         │
│                 │ Translation Result:                     │
│                 │ ┌─────────────────────────────────────┐ │
│                 │ │ Translated text appears here...     │ │
│                 │ │                                     │ │
│                 │ └─────────────────────────────────────┘ │
│                 │ [Copy to Clipboard]                    │
└─────────────────┴───────────────────────────────────────────┘
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection for AI API calls

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ryanma9629/mrct_trans.git
   cd mrct_trans
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file and add your API keys
   DASHSCOPE_API_KEY=sk-your_qwen_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8099`

## 🔧 Configuration

### API Keys Setup

See [API_KEY_Application_Guide.md](API_KEY_Application_Guide.md) for detailed instructions on obtaining API keys.

**Quick setup in `.env` file:**
```bash
# Qwen (Recommended for Chinese)
DASHSCOPE_API_KEY=sk-your_dashscope_key

# DeepSeek (Cost-effective)
DEEPSEEK_API_KEY=your_deepseek_key

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_key

# Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### Technical Dictionary

The included `QS-TB.csv` file contains over 2,700 technical terms for accurate translation of:
- Statistical terms
- Medical research terminology
- Scientific vocabulary
- Clinical trial terminology

Download and customize the dictionary as needed for your specific domain.

## 🎯 Usage

### Web Interface

1. **Select AI Provider**: Choose your preferred LLM provider
2. **Enter API Token**: Paste your API key (auto-filled if configured)
3. **Input Text**: Type or paste text to translate
4. **Translate**: Click translate button or press Ctrl+Enter
5. **Copy Result**: Use the copy button to get translated text

### Language Detection

The app automatically detects input language:
- **English text** → Translates to Chinese
- **Chinese text** → Translates to English  
- **Mixed text** → Translates to English

### API Endpoints

The application provides REST API endpoints for programmatic access:

```python
# Translation endpoint
POST /translate
{
  "text": "Hello world",
  "llm_provider": "qwen",
  "api_token": "your-token",
  "model": "qwen-turbo"
}

# Configuration endpoint
GET /config

# Dictionary download
GET /download-dictionary
```

### Translation Service API

You can also use the translation service directly in Python:

```python
from translator import TranslationService

# Initialize service
service = TranslationService()

# Translate text
translated_text, matches = await service.translate(
    text="Hello world",
    llm_provider="qwen",
    api_token="your-token",
    model="qwen-turbo"
)

# Get configuration
config = service.get_config()
```

## 📁 Project Structure

```
mrct_trans/
├── main.py                     # FastAPI web server and API endpoints
├── translator.py               # Translation service core logic
├── static/                     # Frontend files
│   ├── index.html             # Web interface
│   ├── style.css              # Styling
│   └── script.js              # JavaScript logic
├── QS-TB.csv                  # Technical dictionary (2,700+ terms)
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── README.md                 # This file
├── INSTALLATION.md           # Detailed setup guide
└── API_KEY_Application_Guide.md # API key instructions
```

## 🏗️ Architecture

The application follows a clean architecture pattern with separation of concerns:

- **`main.py`**: FastAPI web framework handling HTTP requests and responses
- **`translator.py`**: Core translation service containing:
  - `TranslationService`: Main service class coordinating all translation operations
  - `DictionaryMatcher`: Technical dictionary lookup and context preparation
  - LLM provider integrations (ChatGPT, Azure OpenAI, DeepSeek, Qwen)
  - Language detection and text processing
- **`static/`**: Frontend web interface for user interaction

## 🛡️ Security & Privacy

- ✅ **Local Processing**: All operations run on your machine
- ✅ **No Data Collection**: No user data is stored or transmitted
- ✅ **API Key Protection**: Keys stored locally in `.env` file
- ✅ **Direct API Calls**: Translation requests go directly to chosen AI provider
- ⚠️ **Keep Keys Secure**: Never commit `.env` file to version control

## 🚨 Troubleshooting

### Common Issues

**"Module not found" error**
```bash
# Make sure you're in the virtual environment
pip install -r requirements.txt
```

**API key errors**
- Verify your API key is correctly copied
- Check that the AI service is activated
- Ensure sufficient account balance

**Port 8099 in use**
```bash
# Change port in main.py or kill existing process
# Windows: netstat -ano | findstr :8099
# Linux/Mac: lsof -i :8099
```

**Translation failures**
- Check internet connection
- Verify API provider status
- Try a different AI provider

### Getting Help

1. Check [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions
2. Review [API_KEY_Application_Guide.md](API_KEY_Application_Guide.md) for API setup
3. Open an issue on GitHub with error details and system info

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional AI provider integrations
- Enhanced dictionary management
- UI/UX improvements
- Translation quality optimizations
- Docker containerization

## 📊 Performance

- **Translation Speed**: ~1-3 seconds depending on provider
- **Dictionary Size**: 2,700+ technical terms
- **Memory Usage**: ~50MB typical
- **Supported Text Length**: Up to 4,000 tokens per request

## 🛣️ Roadmap

- [ ] Batch translation support
- [ ] Translation history
- [ ] Custom dictionary management
- [ ] Offline translation option
- [ ] Mobile-responsive improvements
- [ ] Translation quality metrics

## 📄 License

This project is for educational and research purposes. Please check individual AI provider terms of service for commercial usage restrictions.

## 🙏 Acknowledgments

- Technical dictionary sourced from QS-TB terminology database
- Built with FastAPI and modern web technologies
- Supports multiple leading AI translation services

---

**Made with ❤️ for accurate English ⇄ Chinese translation**

*For technical support or questions, please refer to the documentation or open an issue.*