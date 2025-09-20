# Technical Translation Tool

A web application for English ⇄ Chinese translation using LLMs with integrated technical dictionary support.

## Features

- **Smart Translation Direction**:
  - English input → Chinese translation
  - Chinese or mixed Chinese/English input → English translation

- **LLM Support**:
  - ChatGPT (OpenAI)
  - ChatGPT (Azure OpenAI)
  - DeepSeek
  - Qwen

- **Technical Dictionary Integration**:
  - Uses QS-TB.csv for technical term translations
  - Handles overlapping terms (longer phrases take precedence)
  - Highlights dictionary matches with underlines

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys** in `.env`:
   ```bash
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key
   
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
   
   # DeepSeek
   DEEPSEEK_API_KEY=your_deepseek_key
   
   # Qwen
   QWEN_API_KEY=your_qwen_key
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

4. **Open browser** to `http://localhost:8000`

## Usage

1. Select your preferred LLM provider
2. Enter your API token (auto-filled if configured in .env)
3. Type or paste text to translate
4. Click "Translate" or press Ctrl+Enter
5. Copy results with the "Copy to Clipboard" button

## Dictionary Format

The QS-TB.csv file should have:
- Column 1: English terms
- Column 2: Chinese translations
- First row: headers (EN,CN,...)

Technical terms from the dictionary will be highlighted with underlines in the translation output.