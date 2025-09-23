# API Key Application Guide

This document provides instructions on how to obtain API keys for Alibaba Qwen, DeepSeek, OpenAI (ChatGPT), and Azure OpenAI for use with the MRCT BOOK Translator tool.

---

## Alibaba Qwen API Key Application

### Step 1: Register Alibaba Cloud Account
1. Visit the Alibaba Cloud website: https://www.aliyun.com/
2. Register for an account and complete the real-name verification.

### Step 2: Activate Model Studio (Bailian)
1. Log into the Alibaba Cloud console and search for "Model Studio" (灵积模型服务).
2. Visit the Model Studio console: https://bailian.console.aliyun.com/
3. Activate the service and agree to the terms.

### Step 3: Get API Key
1. In the Model Studio console, navigate to "API Key Management".
2. Click "Create New API Key" and copy the generated key (format: `sk-xxxxxxxxxx`).

### Usage Information
- **Environment Variable**: `DASHSCOPE_API_KEY`
- **Default Model**: `qwen-turbo`
- **Documentation**: https://help.aliyun.com/zh/model-studio/

---

## DeepSeek API Key Application

### Step 1: Register DeepSeek Account
1. Visit the DeepSeek platform: https://platform.deepseek.com/
2. Sign up with your email or a GitHub account.

### Step 2: Get API Key
1. After logging in, go to the "API Keys" page.
2. Click "Create new secret key" and copy the generated API Key.

### Usage Information
- **Environment Variable**: `DEEPSEEK_API_KEY`
- **Default Model**: `deepseek-chat`
- **Documentation**: https://platform.deepseek.com/docs

---

## OpenAI (ChatGPT) API Key Application

### Step 1: Register OpenAI Account
1. Visit the OpenAI Platform: https://platform.openai.com/
2. Sign up for a new account and verify your email.

### Step 2: Get API Key
1. Navigate to the "API keys" section in your account dashboard.
2. Click "Create new secret key" and copy the generated key.

### Step 3: Set Up Billing
1. Go to the "Billing" section and add a payment method. A paid plan is required to access newer models and higher rate limits.

### Usage Information
- **Environment Variable**: `OPENAI_API_KEY`
- **Default Model**: `gpt-4o-mini`
- **Documentation**: https://platform.openai.com/docs

---

## Azure OpenAI API Key Application

### Step 1: Apply for Azure OpenAI Access
1. You need an Azure subscription. If you don't have one, create one at https://azure.microsoft.com/.
2. Apply for access to the Azure OpenAI Service by filling out the request form. Access is currently limited.

### Step 2: Create and Deploy an OpenAI Resource
1. Once approved, create an "Azure OpenAI" resource in the Azure portal.
2. In your resource, go to "Model deployments" and deploy a model (e.g., `gpt-4o-mini`). You must give your deployment a unique name.

### Step 3: Get API Key and Endpoint
1. Navigate to the "Keys and Endpoint" section of your Azure OpenAI resource.
2. Copy the **API Key** and the **Endpoint URL**.

### Usage Information
- **Environment Variables**:
  - `AZURE_OPENAI_API_KEY`: Your API key.
  - `AZURE_OPENAI_ENDPOINT`: Your endpoint URL.
- **Model**: The **deployment name** you created in Step 2.

---

## Configuration Recommendations

### Environment Variable Setup
Create a `.env` file in the project root with the following format:
```bash
# Qwen API Configuration
DASHSCOPE_API_KEY=sk-your_qwen_api_key

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key

# OpenAI (ChatGPT) API Configuration
OPENAI_API_KEY=your_openai_api_key

# Azure OpenAI API Configuration
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-azure-endpoint.openai.azure.com/
```

### Security Notes
- **Never expose API Keys**: Do not commit your `.env` file or keys to public repositories.
- **Monitor Usage**: Set reasonable usage limits and alerts in your provider's dashboard.
- **Minimal Permissions**: Grant only the necessary permissions for your API keys.

### Cost Control
- **Start Small**: Begin with a small credit or low spending limit for testing.
- **Check Pricing**: Review the pricing for each model on the respective provider's website, as costs can vary significantly.
- **Qwen & DeepSeek**: Generally more cost-effective, especially for Chinese translations.
- **OpenAI & Azure**: Typically more expensive but offer high-quality models.

---

## Troubleshooting

If you encounter an **Invalid API Key** error:
- Ensure the key is copied correctly.
- Confirm the corresponding service is activated and has a valid payment method.
- Check your account balance or spending limits.
- For Azure, ensure you are using the correct endpoint and deployment name.

---

## Update Log
- 2025-09-23: Added guides for OpenAI and Azure OpenAI. Updated structure.
- 2025-09-20: Document created.
- Document Version: v2.0