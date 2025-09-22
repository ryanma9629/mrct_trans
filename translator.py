import os
import csv
import re
import logging
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RAG_CONTEXTS = 1
MAX_TOKENS_WITH_RAG = 3000
TECHNICAL_TRANSLATION_TEMPERATURE = 0.1
LOG_PREVIEW_CHARS = 100
CONTEXT_PREVIEW_CHARS = 150


class SupportedProvider(Enum):
    CHATGPT = "chatgpt"
    CHATGPT_AZURE = "chatgpt(azure)"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


@dataclass
class ProviderConfig:
    endpoint: str
    default_model: str
    max_tokens: int
    timeout: Optional[float] = None


class RAGConfig:
    """Configuration class for RAG components."""
    
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    @property
    def is_complete(self) -> bool:
        """Check if all required RAG configuration is available."""
        return all([
            self.azure_endpoint,
            self.api_key,
            self.api_version,
            self.embedding_deployment
        ])
    
    def log_status(self) -> None:
        """Log the configuration status for debugging."""
        logger.debug(f"Azure OpenAI config check - "
                    f"Endpoint: {'✓' if self.azure_endpoint else '✗'}, "
                    f"API Key: {'✓' if self.api_key else '✗'}, "
                    f"API Version: {'✓' if self.api_version else '✗'}, "
                    f"Embedding Deployment: {'✓' if self.embedding_deployment else '✗'}")


class TranslationError(Exception):
    pass


class APIError(TranslationError):
    pass


class DictionaryMatcher:
    
    def __init__(self, csv_file: Union[str, Path]):
        self.en_to_cn: Dict[str, str] = {}
        self.cn_to_en: Dict[str, str] = {}
        self._load_dictionary(csv_file)
    
    def _load_dictionary(self, csv_file: Union[str, Path]) -> None:
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)
                
                for row_num, row in enumerate(reader, start=2):
                    if len(row) >= 2 and row[0].strip() and row[1].strip():
                        en_term = row[0].strip().lower()
                        cn_term = row[1].strip()
                        
                        if en_term not in self.en_to_cn:
                            self.en_to_cn[en_term] = cn_term
                        
                        if cn_term not in self.cn_to_en:
                            self.cn_to_en[cn_term] = row[0].strip()
                    elif len(row) >= 2:
                        logger.debug(f"Skipping empty entry at row {row_num}: {row}")
                        
            logger.info(f"Loaded {len(self.en_to_cn)} dictionary entries from {csv_path}")
            
        except (IOError, csv.Error) as e:
            raise TranslationError(f"Failed to load dictionary from {csv_path}: {e}")
    
    def find_matches(self, text: str, is_english: bool) -> List[Tuple[str, str, int, int]]:
        if not text.strip():
            return []
            
        matches = []
        dictionary = self.en_to_cn if is_english else self.cn_to_en
        
        if not dictionary:
            logger.warning("Dictionary is empty")
            return []
        
        sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)
        
        text_lower = text.lower() if is_english else text
        used_positions = set()
        
        for term in sorted_terms:
            try:
                if is_english:
                    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                    search_text = text_lower
                else:
                    pattern = re.compile(re.escape(term))
                    search_text = text
                
                for match in pattern.finditer(search_text):
                    start, end = match.span()
                    if not any(pos in used_positions for pos in range(start, end)):
                        matches.append((term, dictionary[term], start, end))
                        used_positions.update(range(start, end))
                        
            except re.error as e:
                logger.warning(f"Regex error for term '{term}': {e}")
                continue
        
        return sorted(matches, key=lambda x: x[2])


class TranslationService:
    
    _PROVIDER_CONFIGS = {
        SupportedProvider.DEEPSEEK: ProviderConfig(
            endpoint=os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/chat/completions"),
            default_model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            max_tokens=4000,
            timeout=60.0
        ),
        SupportedProvider.QWEN: ProviderConfig(
            endpoint=os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            default_model=os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo"),
            max_tokens=2000,
            timeout=None
        )
    }
    
    def __init__(self, dictionary_file: Union[str, Path] = "QS-TB.csv", 
                 chroma_db_path: str = "chroma_db"):
        try:
            self.dictionary_matcher = DictionaryMatcher(dictionary_file)
        except (FileNotFoundError, TranslationError) as e:
            logger.error(f"Failed to initialize dictionary matcher: {e}")
            raise
        
        # RAG components
        self.chroma_db_path = chroma_db_path
        self.vectorstore = None
        self.embeddings = None
        self._initialize_rag_components()
    
    def _initialize_rag_components(self) -> None:
        """Initialize ChromaDB connection and Azure OpenAI embeddings for RAG."""
        logger.info("Initializing RAG components...")
        
        try:
            # Check configuration
            rag_config = RAGConfig()
            rag_config.log_status()
            
            if not rag_config.is_complete:
                self._disable_rag_with_message("Azure OpenAI configuration for embeddings not found")
                return
            
            # Initialize embeddings
            self._initialize_embeddings(rag_config)
            
            # Initialize vector store
            self._initialize_vector_store()
            
            logger.info("RAG components initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            logger.debug(f"RAG initialization error details: {type(e).__name__}: {str(e)}")
            self._disable_rag_with_message("Check your configuration and try again")
    
    def _disable_rag_with_message(self, additional_message: str) -> None:
        """Disable RAG components and log informative message."""
        logger.info(f"RAG features will be disabled. {additional_message}")
        logger.info("Required environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                   "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.vectorstore = None
        self.embeddings = None
    
    def _initialize_embeddings(self, rag_config: RAGConfig) -> None:
        """Initialize Azure OpenAI embeddings."""
        # Set environment variable for langchain (we know api_key is not None due to is_complete check)
        if rag_config.api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = rag_config.api_key
            logger.debug("Azure OpenAI API key set for embeddings")
        
        logger.info(f"Initializing Azure OpenAI embeddings with deployment: {rag_config.embedding_deployment}")
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=rag_config.azure_endpoint,
            azure_deployment=rag_config.embedding_deployment,
            api_version=rag_config.api_version,
        )
        logger.info("Azure OpenAI embeddings initialized successfully")
    
    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB vector store."""
        logger.info(f"Checking ChromaDB directory: {self.chroma_db_path}")
        
        if not os.path.exists(self.chroma_db_path):
            logger.warning(f"ChromaDB directory not found: {self.chroma_db_path}")
            self._disable_rag_with_message("Run load_pdfs.py to enable RAG")
            logger.info("To populate the database: python load_pdfs.py")
            return
        
        logger.info("Initializing ChromaDB vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings
        )
        
        # Test the vector store
        self._test_vector_store()
    
    def _test_vector_store(self) -> None:
        """Test vector store and log document count."""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'get'):
                collection_count = len(self.vectorstore.get()['ids'])
                logger.info(f"ChromaDB vector store initialized successfully with {collection_count} documents")
            else:
                logger.info("ChromaDB vector store initialized successfully")
        except Exception as e:
            logger.debug(f"Could not get document count: {e}")
            logger.info("ChromaDB vector store initialized successfully")
    
    def _retrieve_context(self, text: str, chapter_number: int, 
                         num_contexts: int = DEFAULT_RAG_CONTEXTS) -> List[str]:
        """
        Retrieve relevant context from the vector database.
        Only retrieves from the specified chapter. Returns empty list if no context found.
        
        Args:
            text: The text to find similar content for
            chapter_number: Chapter number to filter results (required)
            num_contexts: Number of context chunks to retrieve (default: 1)
            
        Returns:
            List of relevant text chunks (max num_contexts, empty if none found)
        """
        if not self.vectorstore:
            logger.warning("Vector store not initialized, context retrieval skipped")
            return []
        
        try:
            logger.info(f"Starting context retrieval for Chapter {chapter_number} (max {num_contexts} contexts)")
            logger.debug(f"Query text (first {LOG_PREVIEW_CHARS} chars): {text[:LOG_PREVIEW_CHARS]}...")
            
            # Retrieve documents using metadata filtering or fallback
            docs = self._search_with_chapter_filter(text, chapter_number, num_contexts)
            
            if not docs:
                logger.warning(f"No documents found for Chapter {chapter_number}. "
                              f"Either Chapter{chapter_number}.pdf was not loaded or metadata filtering failed.")
                return []
            
            # Extract contexts and log details
            contexts = self._extract_contexts_from_docs(docs, chapter_number)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            logger.error("Traceback:", exc_info=True)
            return []
    
    def _search_with_chapter_filter(self, text: str, chapter_number: int, num_contexts: int) -> List:
        """Search for documents with chapter filtering. Returns empty list if filtering fails."""
        if not self.vectorstore:
            return []
            
        chapter_filter = f"Chapter{chapter_number}"
        logger.debug(f"Filtering documents by source containing: {chapter_filter}")
        
        try:
            # Try ChromaDB metadata filtering
            metadata_filter = {"source": chapter_filter}
            docs = self.vectorstore.similarity_search(
                text,
                k=num_contexts,
                filter=metadata_filter
            )
            logger.debug(f"Metadata filtering successful - found {len(docs)} documents")
            return docs
            
        except Exception as filter_error:
            logger.warning(f"Metadata filtering failed: {filter_error}")
            logger.info("Falling back to non-RAG translation")
            return []  # Return empty list to trigger non-RAG fallback
    
    def _extract_contexts_from_docs(self, docs: List, chapter_number: int) -> List[str]:
        """Extract context strings from documents and log details."""
        if not docs:
            logger.info(f"No documents found for Chapter {chapter_number}")
            return []
            
        logger.info(f"Found {len(docs)} documents from Chapter {chapter_number}")
        
        contexts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '')
            content = doc.page_content
            
            logger.debug(f"Document {i+1}: source='{source}', content_length={len(content)}")
            logger.debug(f"Context {i+1} preview (first {CONTEXT_PREVIEW_CHARS} chars): {content[:CONTEXT_PREVIEW_CHARS]}...")
            contexts.append(content)
        
        total_context_length = sum(len(ctx) for ctx in contexts)
        logger.info(f"Context retrieval completed: {len(contexts)} contexts from Chapter {chapter_number}")
        logger.debug(f"Total context length: {total_context_length} characters")
        
        return contexts

    def detect_language(self, text: str) -> bool:
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        non_space_chars = text.replace(' ', '')
        
        if not non_space_chars:
            raise ValueError("Text contains only spaces")
            
        chinese_ratio = len(chinese_chars) / len(non_space_chars)
        return chinese_ratio < 0.1
    
    def _prepare_dictionary_context(self, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        if not dictionary_matches:
            return ""
        
        dict_context = "\n\nTECHNICAL TERMINOLOGY: The following specialized terms must be translated exactly as specified:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nThese are established technical translations that must be used consistently. Translate all other content while preserving the exact terminology above."
        
        return dict_context
    
    def _prepare_system_prompt(self, is_english: bool, dictionary_matches: Optional[List[Tuple[str, str, int, int]]], 
                              contexts: Optional[List[str]] = None) -> str:
        direction = "from English to Chinese" if is_english else "from Chinese to English"
        dict_context = self._prepare_dictionary_context(dictionary_matches)
        
        base_prompt = (
            f"You are a professional technical and scientific document translator specializing in {direction} translation. "
            f"Your task is to provide accurate, precise, and consistent translations that maintain the technical rigor of the original text.\n\n"
            f"Guidelines:\n"
            f"- Preserve all technical terminology, scientific concepts, and numerical data exactly\n"
            f"- Maintain the formal academic tone and structure of the original document\n"
            f"- Keep abbreviations, formulas, citations, and references unchanged\n"
            f"- Ensure consistency in terminology throughout the translation\n"
            f"- Do not add explanations, interpretations, or additional content\n"
            f"- Provide only the direct, professional translation of the input text"
        )
        
        # Add context if available (RAG)
        if contexts:
            context_text = "\n\n".join(contexts)
            total_context_length = len(context_text)
            context_prompt = (
                f"\n\nRELEVANT CONTEXT: The following text provides context from the same book to help you understand "
                f"the subject matter and maintain consistency. This context is for reference only - "
                f"DO NOT translate this context, only use it to inform your translation:\n\n"
                f"--- CONTEXT START ---\n"
                f"{context_text}\n"
                f"--- CONTEXT END ---\n\n"
                f"Remember: Only translate the user's submitted text, not the context above."
            )
            base_prompt += context_prompt
            logger.debug(f"Enhanced prompt with {len(contexts)} context chunks ({total_context_length} chars)")
        else:
            logger.debug("No context available - using standard prompt")
        
        return base_prompt + dict_context
    
    def _validate_provider(self, provider: str) -> SupportedProvider:
        try:
            return SupportedProvider(provider)
        except ValueError:
            supported_providers = [p.value for p in SupportedProvider]
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: {supported_providers}")
    
    async def _call_llm_api(self, text: str, api_token: str, is_english: bool, 
                           dictionary_matches: Optional[List[Tuple[str, str, int, int]]], 
                           provider: str, model: Optional[str] = None,
                           contexts: Optional[List[str]] = None) -> str:
        provider_enum = self._validate_provider(provider)
        system_prompt = self._prepare_system_prompt(is_english, dictionary_matches, contexts)
        
        if provider_enum in [SupportedProvider.CHATGPT, SupportedProvider.CHATGPT_AZURE]:
            is_azure = provider_enum == SupportedProvider.CHATGPT_AZURE
            return await self._call_openai_api(text, api_token, system_prompt, is_azure, model)
        else:
            return await self._call_http_api(text, api_token, system_prompt, provider_enum, model)
    
    async def _call_openai_api(self, text: str, api_token: str, system_prompt: str, 
                              is_azure: bool, model: Optional[str]) -> str:
        try:
            if is_azure:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if not azure_endpoint:
                    raise APIError("AZURE_OPENAI_ENDPOINT not configured")
                
                client = AsyncAzureOpenAI(
                    api_key=api_token,
                    azure_endpoint=azure_endpoint,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
                )
                default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
                selected_model = model if model else default_model
                
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate: {text}"}
                    ],
                    max_tokens=MAX_TOKENS_WITH_RAG,  # Increased for RAG contexts
                    temperature=TECHNICAL_TRANSLATION_TEMPERATURE
                )
            else:
                client = AsyncOpenAI(api_key=api_token)
                default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
                selected_model = model if model else default_model
                
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate: {text}"}
                    ],
                    max_tokens=MAX_TOKENS_WITH_RAG,  # Increased for RAG contexts
                    temperature=TECHNICAL_TRANSLATION_TEMPERATURE
                )
            
            if not response.choices or not response.choices[0].message.content:
                raise APIError("Empty response from OpenAI API")
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            provider_name = "Azure OpenAI" if is_azure else "OpenAI"
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider_name} API error: {e}")
    
    async def _call_http_api(self, text: str, api_token: str, system_prompt: str, 
                            provider: SupportedProvider, model: Optional[str]) -> str:
        if provider not in self._PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported HTTP provider: {provider}")
        
        config = self._PROVIDER_CONFIGS[provider]
        selected_model = model if model else config.default_model
        
        request_data = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate: {text}"}
            ],
            "max_tokens": min(config.max_tokens + 1000, 6000),  # Increased for RAG contexts
            "temperature": TECHNICAL_TRANSLATION_TEMPERATURE
        }
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            if config.timeout:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(config.endpoint, headers=headers, json=request_data)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(config.endpoint, headers=headers, json=request_data)
            
            if response.status_code != 200:
                await self._handle_http_error(response, provider)
            
            result = response.json()
            
            if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
                raise APIError(f"Empty response from {provider.value} API")
                
            return result["choices"][0]["message"]["content"].strip()
            
        except httpx.RequestError as e:
            raise APIError(f"Network error calling {provider.value} API: {e}")
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"{provider.value} API error: {e}")
    
    async def _handle_http_error(self, response: httpx.Response, provider: SupportedProvider) -> None:
        error_detail = f"{provider.value} API error: {response.status_code}"
        
        try:
            error_body = response.json()
            if "error" in error_body:
                error_message = error_body["error"].get("message", "Unknown error")
                error_detail += f" - {error_message}"
        except Exception:
            error_detail += f" - Response: {response.text[:200]}"
        
        raise APIError(error_detail)
    
    async def translate(self, text: str, llm_provider: str, api_token: str, 
                       model: Optional[str] = None, rag: bool = False,
                       chapter_number: Optional[int] = None) -> Tuple[str, List[Tuple[str, str, int, int]], List[str]]:
        """
        Translate text with optional RAG enhancement.
        
        Args:
            text: Text to translate
            llm_provider: LLM provider to use
            api_token: API token for the provider
            model: Optional model specification
            rag: Whether to use RAG enhancement
            chapter_number: Chapter number for RAG context (required if rag=True)
            
        Returns:
            Tuple of (translated_text, dictionary_matches, contexts)
        """
        # Validate inputs
        self._validate_translation_inputs(text, api_token, rag, chapter_number)
        
        try:
            # Prepare translation components
            is_english = self.detect_language(text)
            matches = self.dictionary_matcher.find_matches(text, is_english)
            
            logger.info(f"Translation request - Text: {len(text)} chars, Provider: {llm_provider}, "
                       f"RAG: {rag}, Chapter: {chapter_number}, Dictionary matches: {len(matches)}")
            
            # Handle RAG context retrieval
            contexts, effective_rag = self._handle_rag_context(rag, text, chapter_number)
            
            logger.info(f"Proceeding with translation: RAG={effective_rag}, contexts={len(contexts)}, "
                       f"matches={len(matches)}")
            
            # Perform translation
            translated = await self._call_llm_api(
                text, api_token, is_english, matches, llm_provider, model, contexts
            )
            
            if not translated.strip():
                raise TranslationError("Translation result is empty")
            
            # Log completion summary
            self._log_translation_summary(len(text), len(translated), effective_rag, 
                                        len(contexts), len(matches), rag, chapter_number)
            
            return translated, matches, contexts
            
        except (ValueError, APIError):
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}")
    
    def _validate_translation_inputs(self, text: str, api_token: str, rag: bool, 
                                   chapter_number: Optional[int]) -> None:
        """Validate translation input parameters."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not api_token.strip():
            raise ValueError("API token cannot be empty")
        
        if rag and chapter_number is None:
            raise ValueError("Chapter number is required when RAG is enabled")
    
    def _handle_rag_context(self, rag: bool, text: str, chapter_number: Optional[int]) -> Tuple[List[str], bool]:
        """
        Handle RAG context retrieval with fallback logic.
        
        Returns:
            Tuple of (contexts, effective_rag_status)
        """
        contexts = []
        effective_rag = rag
        
        if not rag:
            logger.info("RAG disabled - using standard translation without context")
            return contexts, False
        
        if not self.vectorstore:
            logger.warning("RAG requested but vector store not available. Proceeding without RAG.")
            logger.info("To enable RAG: 1) Run load_pdfs.py, 2) Check Azure OpenAI embedding config")
            return contexts, False
        
        if chapter_number is None:
            logger.error("RAG enabled but chapter_number is None - this should have been caught earlier")
            return contexts, False
        
        # Retrieve context
        logger.info(f"RAG enabled - retrieving context for Chapter {chapter_number}")
        contexts = self._retrieve_context(text, chapter_number, num_contexts=DEFAULT_RAG_CONTEXTS)
        
        if contexts:
            logger.info(f"RAG context successfully retrieved: {len(contexts)} chunks")
        else:
            logger.warning(f"RAG enabled but no context found for Chapter {chapter_number}")
            logger.info("Falling back to standard translation without RAG context")
            effective_rag = False
        
        return contexts, effective_rag
    
    def _log_translation_summary(self, input_length: int, output_length: int, 
                               effective_rag: bool, context_count: int, match_count: int,
                               requested_rag: bool, chapter_number: Optional[int]) -> None:
        """Log detailed translation completion summary."""
        logger.info("Translation completed successfully")
        logger.info(f"Summary - Input: {input_length} chars, Output: {output_length} chars, "
                   f"RAG: {effective_rag}, Contexts: {context_count}, Dict matches: {match_count}")
        
        if effective_rag and context_count > 0:
            logger.info(f"RAG enhanced translation completed with {context_count} context chunks from Chapter {chapter_number}")
        elif requested_rag and not context_count:
            logger.info(f"RAG was requested but no contexts found for Chapter {chapter_number} - used standard translation")
        else:
            logger.info("Standard translation completed without RAG context")
    
    def get_config(self) -> Dict[str, Union[List[str], Dict[str, str]]]:
        return {
            "supported_providers": [provider.value for provider in SupportedProvider],
            "default_models": {
                SupportedProvider.CHATGPT.value: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                SupportedProvider.CHATGPT_AZURE.value: os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                SupportedProvider.DEEPSEEK.value: os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
                SupportedProvider.QWEN.value: os.getenv("QWEN_DEFAULT_MODEL", "qwen-turbo")
            },
            "default_tokens": {
                SupportedProvider.CHATGPT.value: os.getenv("OPENAI_API_KEY", ""),
                SupportedProvider.CHATGPT_AZURE.value: os.getenv("AZURE_OPENAI_API_KEY", ""),
                SupportedProvider.DEEPSEEK.value: os.getenv("DEEPSEEK_API_KEY", ""),
                SupportedProvider.QWEN.value: os.getenv("DASHSCOPE_API_KEY", "")
            }
        }