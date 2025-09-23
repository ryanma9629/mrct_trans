import os
import csv
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dotenv import load_dotenv
from difflib import SequenceMatcher

from llm import LLMService, APIError, SupportedProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.618
CONTEXT_WINDOW_EXPAND_LENGTH = 1000
TECHNICAL_TRANSLATION_TEMPERATURE = 0.1


class TranslationError(Exception):
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
    
    def __init__(self, dictionary_file: Union[str, Path] = "QS-TB.csv", 
                 context_data_path: str = "context_data"):
        try:
            self.dictionary_matcher = DictionaryMatcher(dictionary_file)
        except (FileNotFoundError, TranslationError) as e:
            logger.error(f"Failed to initialize dictionary matcher: {e}")
            raise
        
        self.llm_service = LLMService()
        
        # Text-based context components
        self.context_data_path = context_data_path
        self.context_enabled = self._check_context_data_path()

    def _check_context_data_path(self) -> bool:
        """Check if the context data directory exists and is not empty."""
        if not os.path.isdir(self.context_data_path):
            logger.warning(f"Context data directory not found: '{self.context_data_path}'.")
            logger.info("Context retrieval will be disabled. Run 'python load_pdfs.py' to create it.")
            return False
        
        if not any(fname.endswith('.txt') for fname in os.listdir(self.context_data_path)):
            logger.warning(f"Context data directory '{self.context_data_path}' is empty.")
            logger.info("Context retrieval will be disabled. Run 'python load_pdfs.py' to populate it.")
            return False
            
        logger.info(f"Context data found at '{self.context_data_path}'. Context retrieval is enabled.")
        return True

    def _retrieve_text_context(self, text: str, chapter_number: int, 
                               similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Optional[Tuple[str, float]]:
        """
        Retrieve relevant context from a chapter's text file using string similarity.
        Returns a tuple of (context_text, similarity_score) or None.
        """
        if not self.context_enabled:
            logger.warning("Context retrieval skipped because context data is not available.")
            return None

        chapter_file = f"Chapter{chapter_number}.txt"
        file_path = os.path.join(self.context_data_path, chapter_file)

        if not os.path.exists(file_path):
            logger.warning(f"Context file not found for Chapter {chapter_number}: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read()
        except IOError as e:
            logger.error(f"Could not read context file {file_path}: {e}")
            return None

        logger.info(f"Searching for best match in Chapter {chapter_number}...")
        
        # Normalize both input and chapter text to handle line break differences
        normalized_input = re.sub(r'\s+', ' ', text).strip()
        normalized_chapter = re.sub(r'\s+', ' ', chapter_text).strip()

        # Use SequenceMatcher on the normalized text
        matcher = SequenceMatcher(None, normalized_input, normalized_chapter, autojunk=False)
        match = matcher.find_longest_match(0, len(normalized_input), 0, len(normalized_chapter))

        if match.size == 0 or len(normalized_input) == 0:
            logger.warning("No match found in the chapter text after normalization.")
            return None

        similarity = match.size / len(normalized_input)
        
        # To find the original block, we can use the match indices on the normalized string
        # and then map them back to the original chapter text. This is an approximation.
        # A more robust way is to find the start of the matched block in the original text.
        
        # Get the actual text that was matched from the normalized chapter
        matched_block_normalized = normalized_chapter[match.b : match.b + match.size]
        
        # To find this block in the original, non-normalized text, we can create a regex
        # that is flexible with whitespace.
        # Escape special regex characters in the matched text and replace spaces with '\s+'
        # to match any whitespace sequence.
        regex_pattern = re.escape(matched_block_normalized).replace(r'\ ', r'\s+')
        
        try:
            original_match = re.search(regex_pattern, chapter_text)
            if not original_match:
                logger.warning("Could not map normalized match back to original text. Using fallback.")
                # Fallback to using the normalized match indices directly (less accurate)
                original_start_index = match.b
                original_end_index = match.b + match.size
            else:
                original_start_index = original_match.start()
                original_end_index = original_match.end()

        except re.error:
            logger.error("Regex error during context mapping. Using fallback.")
            original_start_index = match.b
            original_end_index = match.b + match.size

        # Expand the context window by up to CONTEXT_WINDOW_EXPAND_LENGTH characters before and after the match.
        context_start = max(0, original_start_index - CONTEXT_WINDOW_EXPAND_LENGTH)
        context_end = min(len(chapter_text), original_end_index + CONTEXT_WINDOW_EXPAND_LENGTH)
        best_match_context = chapter_text[context_start:context_end]
        
        logger.info(f"Found best match with similarity: {similarity:.4f} (threshold: {similarity_threshold})")
        logger.info(f"Retrieved context from {chapter_file}:")
        logger.info("--- CONTEXT START ---")
        logger.info(best_match_context)
        logger.info("--- CONTEXT END ---")

        if similarity >= similarity_threshold:
            logger.info("Similarity is above threshold. Using this context for translation.")
            return best_match_context, similarity
        else:
            logger.warning("Match similarity is below threshold. Context will not be used for translation.")
            return None

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
                              context: Optional[str] = None) -> str:
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
        
        # Add context if available
        if context:
            context_prompt = (
                f"\n\nRELEVANT CONTEXT: The following text provides context from the same book to help you understand "
                f"the subject matter and maintain consistency. This context is for reference only - "
                f"DO NOT translate this context, only use it to inform your translation:\n\n"
                f"--- CONTEXT START ---\n"
                f"{context}\n"
                f"--- CONTEXT END ---\n\n"
                f"Remember: Only translate the user's submitted text, not the context above."
            )
            base_prompt += context_prompt
            logger.debug(f"Enhanced prompt with context ({len(context)} chars)")
        else:
            logger.debug("No context available - using standard prompt")
        
        return base_prompt + dict_context
    
    async def translate(self, text: str, llm_provider: str, api_token: str, 
                       model: Optional[str] = None, use_context: bool = False,
                       chapter_number: Optional[int] = None) -> Tuple[str, List[Tuple[str, str, int, int]], Optional[Dict[str, Union[str, float]]]]:
        """
        Translate text with optional text-based context enhancement.
        
        Args:
            text: Text to translate
            llm_provider: LLM provider to use
            api_token: API token for the provider
            model: Optional model specification
            use_context: Whether to use text-based context enhancement
            chapter_number: Chapter number for context (required if use_context=True)
            
        Returns:
            Tuple of (translated_text, dictionary_matches, context_info)
            where context_info is a dict {'text': str, 'score': float} or None.
        """
        # Validate inputs
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not api_token.strip():
            raise ValueError("API token cannot be empty")
        
        if use_context and chapter_number is None:
            raise ValueError("Chapter number is required when context is enabled")
        
        try:
            # Prepare translation components
            is_english = self.detect_language(text)
            matches = self.dictionary_matcher.find_matches(text, is_english)
            
            logger.info(f"Translation request - Text: {len(text)} chars, Provider: {llm_provider}, "
                       f"Use Context: {use_context}, Chapter: {chapter_number}, Dictionary matches: {len(matches)}")
            
            # Handle context retrieval
            context_info = None
            if use_context:
                if not self.context_enabled:
                    logger.warning("Context requested but not available. Proceeding without context.")
                elif chapter_number is not None:
                    logger.info(f"Context enabled - retrieving context for Chapter {chapter_number}")
                    context_result = self._retrieve_text_context(text, chapter_number)
                    if context_result:
                        context, similarity = context_result
                        logger.info(f"Context successfully retrieved for Chapter {chapter_number} with similarity {similarity:.4f}")
                        context_info = {"text": context, "score": similarity}
                    else:
                        logger.warning(f"Context requested but no suitable context found for Chapter {chapter_number}")

            context_text = context_info['text'] if context_info else None
            
            logger.info(f"Proceeding with translation: Context provided={context_text is not None}, "
                       f"matches={len(matches)}")
            
            # Perform translation
            system_prompt = self._prepare_system_prompt(is_english, matches, context_text)
            translated = await self.llm_service.call_llm_api(
                text=text,
                system_prompt=system_prompt,
                provider=llm_provider,
                api_token=api_token,
                model=model,
                temperature=TECHNICAL_TRANSLATION_TEMPERATURE
            )
            
            if not translated.strip():
                raise TranslationError("Translation result is empty")
            
            # Log completion summary
            logger.info("Translation completed successfully")
            logger.info(f"Summary - Input: {len(text)} chars, Output: {len(translated)} chars, "
                       f"Context Used: {context_text is not None}, Dict matches: {len(matches)}")
            
            if context_text is not None:
                logger.info(f"Context-enhanced translation completed using text from Chapter {chapter_number}")
            elif use_context and context_text is None:
                logger.info(f"Context was requested but no suitable match found for Chapter {chapter_number} - used standard translation")
            else:
                logger.info("Standard translation completed without context")

            return translated, matches, context_info
            
        except (ValueError, APIError):
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}")
    
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