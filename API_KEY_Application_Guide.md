# API Key Application Guide

This document provides instructions on how to obtain API keys for Alibaba Qwen and DeepSeek for use with the MRCT BOOK Translator tool.

## Alibaba Qwen API Key Application

### Step 1: Register Alibaba Cloud Account
1. Visit Alibaba Cloud website: https://www.aliyun.com/
2. Click "Register" button in the top right corner
3. Fill in phone number, verification code, password and complete registration
4. Complete real-name verification (ID required)

### Step 2: Activate Model Studio Service
1. Log into Alibaba Cloud console
2. Search for "Model Studio" or visit: https://bailian.console.aliyun.com/
3. Click "Activate Now"
4. Read and agree to the service agreement

### Step 3: Get API Key
1. Enter Model Studio console
2. Click "API Key Management" in the left navigation
3. Click "Create New API Key"
4. Set API Key name and description
5. Copy the generated API Key (format: sk-xxxxxxxxxx)

### Step 4: Configure Billing and Limits
1. Recharge your account in the console (recommend small amount for testing)
2. Set consumption limits to avoid unexpected charges
3. Check model pricing: https://help.aliyun.com/zh/model-studio/getting-started/models

### Usage Information
- **Environment Variable**: `DASHSCOPE_API_KEY`
- **Default Model**: `qwen-turbo`
- **API Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- **Documentation**: https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api

---

## DeepSeek API Key Application

### Step 1: Register DeepSeek Account
1. Visit DeepSeek website: https://www.deepseek.com/
2. Click "Sign Up" in the top right corner
3. Register with email or login with GitHub
4. Verify email address

### Step 2: Apply for API Access
1. After login, visit: https://platform.deepseek.com/
2. Complete personal information
3. Wait for approval (usually 1-2 business days)

### Step 3: Get API Key
1. After approval, login to platform
2. Go to "API Keys" page
3. Click "Create new secret key"
4. Set key name and permissions
5. Copy the generated API Key (save it safely, shown only once)

### Step 4: Billing and Configuration
1. Go to "Billing" page to add credits
2. Choose appropriate recharge amount
3. Set usage limits and monitoring

### Usage Information
- **Environment Variable**: `DEEPSEEK_API_KEY`
- **Default Model**: `deepseek-chat`
- **API Endpoint**: `https://api.deepseek.com/chat/completions`
- **Documentation**: https://api-docs.deepseek.com/

---

## Configuration Recommendations

### Environment Variable Setup
Create a `.env` file in the project root:
```bash
# Qwen API Configuration
DASHSCOPE_API_KEY=sk-your_dashscope_api_key_here
QWEN_DEFAULT_MODEL=qwen-turbo

# DeepSeek API Configuration  
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_DEFAULT_MODEL=deepseek-chat
```

### Security Notes
1. **Never expose API Keys**: Don't commit API keys to code repositories
2. **Regular rotation**: Recommend changing API keys periodically
3. **Monitor usage**: Set reasonable usage limits and alerts
4. **Minimal permissions**: Only grant necessary API permissions

### Cost Control Recommendations
1. **Qwen**:
   - Input Tokens: $0.56/1M tokens
   - Output Tokens: $1.68/1M tokens
   - Recommendation: Start with small recharge (¥10-50) for testing

2. **DeepSeek**:
   - Relatively low cost, suitable for heavy usage
   - Recommendation: Start with $5-10 for testing

### Troubleshooting
If you encounter these issues:

**Invalid API Key**
- Check if key is correctly copied
- Confirm service is activated
- Check account balance

**Request Rejected**
- Check if API quota is exhausted
- Verify request format is correct
- Check service status page

**Network Connection Issues**
- Check network connection
- Try using VPN (may be needed in some regions)
- Check firewall settings

---

## Technical Support

### Alibaba Qwen
- Official Documentation: https://help.aliyun.com/zh/model-studio/
- Technical Support: Through Alibaba Cloud ticket system
- Community Forum: https://developer.aliyun.com/

### DeepSeek  
- Official Documentation: https://api-docs.deepseek.com/
- Email Support: api-service@deepseek.com
- Discord Community: https://discord.gg/Tc7c45Zzu5

---

## Update Log
- 2025-01-20: Document created
- Document Version: v1.0