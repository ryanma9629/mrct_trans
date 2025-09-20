# MRCT BOOK Translator - Installation & Usage Guide

A cross-platform web-based application for English ⇄ Chinese translation using multiple LLM providers with integrated technical dictionary support.

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+, CentOS 7+)
- **Memory**: 2GB RAM minimum
- **Network**: Internet connection for AI API calls

## 🚀 Installation Instructions

### Windows Installation

#### Method 1: Using Command Prompt
```cmd
# 1. Check Python installation
python --version

# 2. Clone or download the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 3. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create configuration file
copy .env.example .env
```

#### Method 2: Using PowerShell
```powershell
# 1. Check Python installation
python --version

# 2. Clone or download the project
git clone https://github.com/ryanma9629/mrct_trans.git
Set-Location mrct_trans

# 3. Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create configuration file
Copy-Item .env.example .env
```

### macOS Installation

```bash
# 1. Check Python installation (install via Homebrew if needed)
python3 --version
# If not installed: brew install python

# 2. Clone or download the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 3. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create configuration file
cp .env.example .env
```

### Linux Installation

#### Ubuntu/Debian
```bash
# 1. Update package list and install Python (if needed)
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# 2. Clone or download the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 3. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create configuration file
cp .env.example .env
```

#### CentOS/RHEL/Fedora
```bash
# 1. Install Python and dependencies (if needed)
# For CentOS/RHEL
sudo yum install python3 python3-pip python3-devel git
# For Fedora
sudo dnf install python3 python3-pip python3-devel git

# 2. Clone or download the project
git clone https://github.com/ryanma9629/mrct_trans.git
cd mrct_trans

# 3. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create configuration file
cp .env.example .env
```

## 🔧 Configuration

### 1. API Keys Setup

Edit the `.env` file with your preferred text editor:

**Windows:**
```cmd
notepad .env
```

**macOS/Linux:**
```bash
nano .env
# or
vim .env
# or
code .env  # if VS Code is installed
```

### 2. Configure Your API Keys

Add your API keys to the `.env` file:

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

### 3. Getting API Keys

See `API_KEY_Application_Guide.md` for detailed instructions on how to obtain API keys for each provider.

## 🎯 Usage Instructions

### Starting the Application

#### Windows
```cmd
# Start the server
python main.py
```

#### macOS/Linux
```bash
# Start the server
python3 main.py
```

The application will start on `http://localhost:8099`. Open your web browser and navigate to this URL to use the translation interface.

### Using the Application

1. **Start the Service**:
   - Run `python main.py` (or `python3 main.py` on macOS/Linux)
   - The server will start on `http://localhost:8099`
   - Open your web browser and navigate to this URL

2. **Select LLM Provider**:
   - Choose from: Qwen, DeepSeek, ChatGPT, or ChatGPT (Azure)
   - Enter your API token (auto-filled if configured in `.env`)

3. **Translate Text**:
   - **English → Chinese**: Enter English text
   - **Chinese → English**: Enter Chinese or mixed Chinese/English text
   - Click "Translate" or press `Ctrl+Enter`

4. **Copy Results**: Use the copy to clipboard button for easy sharing

### Stopping the Application

- **Terminal**: Press `Ctrl+C` in the terminal where the server is running

## 📁 Project Structure

```
mrct_trans/
├── main.py                      # FastAPI backend server
├── static/                      # Web frontend files
│   ├── index.html              # Main web interface
│   ├── style.css               # Styling
│   └── script.js               # Frontend logic
├── QS-TB.csv                   # Technical dictionary
├── .env                        # Configuration file (create from .env.example)
├── .env.example               # Configuration template
├── requirements.txt           # Python dependencies
├── INSTALLATION.md            # This file
└── API_KEY_Application_Guide.md # API key setup guide
```

## 🛠️ Troubleshooting

## Support

### Common Issues

#### Python Not Found
**Windows:**
```cmd
# Install Python from python.org or Microsoft Store
# Make sure to check "Add Python to PATH" during installation
```

**macOS:**
```bash
# Install using Homebrew
brew install python
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### Permission Denied (Linux/macOS)
```bash
# Make sure you have proper permissions to run Python scripts
python3 main.py
```

#### Port 8099 Already in Use
```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8099

# macOS/Linux:
lsof -i :8099

# Kill the process or change the port in main.py
```

#### API Key Issues
1. **Invalid API Key**: Double-check your API key is correctly copied
2. **Service Not Activated**: Ensure the AI service is activated in your account
3. **Insufficient Balance**: Check your account balance for the AI service
4. **Network Issues**: Check your internet connection and firewall settings

#### Dependencies Installation Failed
```bash
# Update pip first
pip install --upgrade pip

# Install dependencies one by one if batch install fails
pip install fastapi
pip install uvicorn[standard]
pip install python-dotenv
pip install openai
pip install httpx
pip install pydantic
pip install python-multipart
```

### Performance Tips

1. **Virtual Environment**: Always use a virtual environment to avoid conflicts
2. **API Provider**: Choose the most suitable API provider for your use case:
   - **Qwen**: Best for Chinese translations
   - **DeepSeek**: Most cost-effective
   - **ChatGPT**: Most reliable but more expensive
3. **Dictionary**: Keep `QS-TB.csv` updated with your technical terms

## 🔒 Security & Privacy

- All processing runs locally on your machine
- Translation requests are only sent to your chosen AI provider
- API keys are stored locally in the `.env` file
- No data is collected or sent to third parties
- Ensure your `.env` file is not shared or committed to version control

## 📝 Support

### Getting Help

1. **Documentation**: Check `INSTALLATION.md` and `API_KEY_Application_Guide.md`
2. **Issues**: Report bugs or request features on the project repository
3. **Configuration**: Verify your `.env` file setup
4. **Logs**: Check the terminal output for error messages

### System Information

When reporting issues, please include:
- Operating system and version
- Python version (`python --version`)
- Error messages from the launcher or terminal
- API provider being used

## 📄 License

This project is for educational and research purposes. Please check individual AI provider terms of service for commercial usage restrictions.

---

**Quick Start Summary:**
1. Install Python 3.8+
2. Clone/download project
3. Create virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add API keys
6. Run: `python main.py`
7. Open browser to `http://localhost:8099`