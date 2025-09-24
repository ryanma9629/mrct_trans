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

# Translation configuration constants
DEFAULT_SIMILARITY_THRESHOLD = 0.2
CONTEXT_WINDOW_EXPAND_LENGTH = 1000
TECHNICAL_TRANSLATION_TEMPERATURE = 0.1
CHINESE_CHAR_THRESHOLD = 0.1  # Ratio threshold for language detection


class TranslationError(Exception):
    pass


class DictionaryMatcher:
    """Handles loading and matching of technical dictionary terms."""

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

            logger.info(f"Loaded {len(self.en_to_cn)} EN→CN dictionary entries from {csv_path}")
            
        except (IOError, csv.Error) as e:
            raise TranslationError(f"Failed to load dictionary from {csv_path}: {e}")
    
    def find_matches(self, text: str, is_english: bool) -> List[Tuple[str, str, int, int]]:
        """Find dictionary matches in text and return their positions."""
        if not text.strip():
            return []

        dictionary = self.en_to_cn if is_english else self.cn_to_en
        if not dictionary:
            logger.warning("Dictionary is empty")
            return []

        return self._find_text_matches(text, dictionary, is_english)

    def _find_text_matches(self, text: str, dictionary: Dict[str, str], is_english: bool) -> List[Tuple[str, str, int, int]]:
        """Internal method to find and return text matches."""
        matches = []
        sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)
        text_lower = text.lower() if is_english else text
        used_positions = set()

        for term in sorted_terms:
            try:
                pattern = self._create_search_pattern(term, is_english)
                search_text = text_lower if is_english else text

                for match in pattern.finditer(search_text):
                    start, end = match.span()
                    if not self._overlaps_with_used_positions(start, end, used_positions):
                        matches.append((term, dictionary[term], start, end))
                        used_positions.update(range(start, end))

            except re.error as e:
                logger.warning(f"Regex error for term '{term}': {e}")
                continue

        return sorted(matches, key=lambda x: x[2])

    def _create_search_pattern(self, term: str, is_english: bool) -> re.Pattern:
        """Create appropriate regex pattern for term matching."""
        if is_english:
            return re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        else:
            return re.compile(re.escape(term))

    def _overlaps_with_used_positions(self, start: int, end: int, used_positions: set) -> bool:
        """Check if the given range overlaps with already used positions."""
        return any(pos in used_positions for pos in range(start, end))


class TranslationService:
    """Main translation service coordinating dictionary matching, context retrieval, and LLM calls."""

    def __init__(self, dictionary_file: Union[str, Path] = "QS-TB.csv",
                 context_data_path: str = "context_data"):
        self.dictionary_matcher = self._init_dictionary_matcher(dictionary_file)
        self.llm_service = LLMService()
        self.context_data_path = context_data_path
        self.context_enabled = self._check_context_data_path()

    def _init_dictionary_matcher(self, dictionary_file: Union[str, Path]) -> DictionaryMatcher:
        """Initialize dictionary matcher with error handling."""
        try:
            return DictionaryMatcher(dictionary_file)
        except (FileNotFoundError, TranslationError) as e:
            logger.error(f"Failed to initialize dictionary matcher: {e}")
            raise

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
        """Retrieve relevant context from chapter text using similarity matching."""
        if not self._validate_context_retrieval(chapter_number):
            return None

        chapter_text = self._load_chapter_text(chapter_number)
        if not chapter_text:
            return None

        logger.info(f"Searching for best match in Chapter {chapter_number}...")

        similarity, context = self._find_best_context_match(text, chapter_text)

        if similarity >= similarity_threshold:
            logger.info(f"Using context with similarity {similarity:.4f} (threshold: {similarity_threshold})")
            self._log_context_details(context, chapter_number)
            return context, similarity
        else:
            logger.warning(f"Match similarity {similarity:.4f} below threshold {similarity_threshold}")
            return None

    def _validate_context_retrieval(self, chapter_number: int) -> bool:
        """Validate that context retrieval is possible."""
        if not self.context_enabled:
            logger.warning("Context retrieval skipped - context data not available")
            return False

        chapter_file = f"Chapter{chapter_number}.txt"
        file_path = os.path.join(self.context_data_path, chapter_file)

        if not os.path.exists(file_path):
            logger.warning(f"Context file not found for Chapter {chapter_number}: {file_path}")
            return False

        return True

    def _load_chapter_text(self, chapter_number: int) -> Optional[str]:
        """Load chapter text from file."""
        chapter_file = f"Chapter{chapter_number}.txt"
        file_path = os.path.join(self.context_data_path, chapter_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            logger.error(f"Could not read context file {file_path}: {e}")
            return None

    def _find_best_context_match(self, text: str, chapter_text: str) -> Tuple[float, str]:
        """Find the best matching context using sequence matching."""
        # Normalize text for better matching
        normalized_input = re.sub(r'\s+', ' ', text).strip()
        normalized_chapter = re.sub(r'\s+', ' ', chapter_text).strip()

        # Find longest common subsequence
        matcher = SequenceMatcher(None, normalized_input, normalized_chapter, autojunk=False)
        match = matcher.find_longest_match(0, len(normalized_input), 0, len(normalized_chapter))

        if match.size == 0 or len(normalized_input) == 0:
            logger.warning("No match found in chapter text")
            return 0.0, ""

        similarity = match.size / len(normalized_input)
        context = self._extract_context_window(chapter_text, match, normalized_chapter)

        return similarity, context

    def _extract_context_window(self, chapter_text: str, match, normalized_chapter: str) -> str:
        """Extract context window around the matched text."""
        # Map normalized match back to original text
        matched_block = normalized_chapter[match.b : match.b + match.size]
        start_idx, end_idx = self._map_to_original_text(matched_block, chapter_text, match.b, match.size)

        # Expand context window
        context_start = max(0, start_idx - CONTEXT_WINDOW_EXPAND_LENGTH)
        context_end = min(len(chapter_text), end_idx + CONTEXT_WINDOW_EXPAND_LENGTH)

        return chapter_text[context_start:context_end]

    def _map_to_original_text(self, matched_block: str, chapter_text: str, fallback_start: int, fallback_size: int) -> Tuple[int, int]:
        """Map normalized match back to original text positions."""
        try:
            # Create flexible regex pattern
            regex_pattern = re.escape(matched_block).replace(r'\ ', r'\s+')
            original_match = re.search(regex_pattern, chapter_text)

            if original_match:
                return original_match.start(), original_match.end()
            else:
                logger.warning("Could not map normalized match to original text")
                return fallback_start, fallback_start + fallback_size

        except re.error:
            logger.error("Regex error during context mapping")
            return fallback_start, fallback_start + fallback_size

    def _log_context_details(self, context: str, chapter_number: int) -> None:
        """Log context details for debugging."""
        logger.info(f"Retrieved context from Chapter{chapter_number}.txt:")
        logger.info("--- CONTEXT START ---")
        logger.info(context)
        logger.info("--- CONTEXT END ---")

    def detect_language(self, text: str) -> bool:
        """Detect if input text is English (True) or Chinese (False)."""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        non_space_chars = text.replace(' ', '')

        if not non_space_chars:
            raise ValueError("Text contains only spaces")

        chinese_ratio = len(chinese_chars) / len(non_space_chars)
        return chinese_ratio < CHINESE_CHAR_THRESHOLD
    
    def _prepare_dictionary_context(self, dictionary_matches: Optional[List[Tuple[str, str, int, int]]]) -> str:
        """Prepare dictionary context for LLM prompt."""
        if not dictionary_matches:
            return ""

        dict_context = "\n\nTECHNICAL TERMINOLOGY: The following specialized terms must be translated exactly as specified:\n"
        for original_term, translated_term, _, _ in dictionary_matches:
            dict_context += f"- '{original_term}' → '{translated_term}'\n"
        dict_context += "\nThese are established technical translations that must be used consistently. Translate all other content while preserving the exact terminology above."

        return dict_context
    
    def _prepare_system_prompt(self, is_english: bool, dictionary_matches: Optional[List[Tuple[str, str, int, int]]],
                              context: Optional[str] = None) -> str:
        """Prepare comprehensive system prompt for LLM translation."""
        direction = "from English to Chinese" if is_english else "from Chinese to English"

        base_prompt = self._build_base_translation_prompt(direction)
        context_prompt = self._build_context_prompt(context) if context else ""
        dict_prompt = self._prepare_dictionary_context(dictionary_matches)

        full_prompt = base_prompt + context_prompt + dict_prompt

        logger.debug(f"Prepared prompt with context: {bool(context)}, dict terms: {len(dictionary_matches) if dictionary_matches else 0}")
        return full_prompt

    def _build_base_translation_prompt(self, direction: str) -> str:
        """Build base translation guidelines prompt."""
        return (
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

    def _build_context_prompt(self, context: str) -> str:
        """Build context section of the prompt."""
        return (
            f"\n\nRELEVANT CONTEXT: The following text provides context from the same book to help you understand "
            f"the subject matter and maintain consistency. This context is for reference only - "
            f"DO NOT translate this context, only use it to inform your translation:\n\n"
            f"--- CONTEXT START ---\n"
            f"{context}\n"
            f"--- CONTEXT END ---\n\n"
            f"Remember: Only translate the user's submitted text, not the context above."
        )
    
    async def translate(self, text: str, llm_provider: str, api_token: str,
                       model: Optional[str] = None, use_context: bool = False,
                       chapter_number: Optional[int] = None) -> Tuple[str, List[Tuple[str, str, int, int]], Optional[Dict[str, Union[str, float]]]]:
        """Translate text with optional context enhancement and dictionary matching."""
        self._validate_translation_inputs(text, api_token, use_context, chapter_number)

        try:
            # Prepare translation components
            is_english = self.detect_language(text)
            matches = self.dictionary_matcher.find_matches(text, is_english)
            context_info = self._retrieve_context_if_requested(text, use_context, chapter_number)

            self._log_translation_request(text, llm_provider, use_context, chapter_number, matches)

            # Perform translation
            translated = await self._perform_translation(
                text, is_english, matches, context_info,
                llm_provider, api_token, model
            )

            self._log_translation_completion(text, translated, context_info, matches, chapter_number, use_context)

            return translated, matches, context_info

        except (ValueError, APIError):
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}")

    def _validate_translation_inputs(self, text: str, api_token: str, use_context: bool, chapter_number: Optional[int]) -> None:
        """Validate translation input parameters."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if not api_token.strip():
            raise ValueError("API token cannot be empty")
        if use_context and chapter_number is None:
            raise ValueError("Chapter number is required when context is enabled")

    def _retrieve_context_if_requested(self, text: str, use_context: bool, chapter_number: Optional[int]) -> Optional[Dict[str, Union[str, float]]]:
        """Retrieve context information if requested and available."""
        if not use_context:
            return None

        if not self.context_enabled:
            logger.warning("Context requested but not available. Proceeding without context.")
            return None

        if chapter_number is not None:
            logger.info(f"Retrieving context for Chapter {chapter_number}")
            context_result = self._retrieve_text_context(text, chapter_number)
            if context_result:
                context, similarity = context_result
                logger.info(f"Context retrieved with similarity {similarity:.4f}")
                return {"text": context, "score": similarity}
            else:
                logger.warning(f"No suitable context found for Chapter {chapter_number}")

        return None

    async def _perform_translation(self, text: str, is_english: bool, matches: List, context_info: Optional[Dict],
                                  llm_provider: str, api_token: str, model: Optional[str]) -> str:
        """Perform the actual LLM translation call."""
        context_text = context_info['text'] if context_info else None
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

        return translated

    def _log_translation_request(self, text: str, llm_provider: str, use_context: bool, chapter_number: Optional[int], matches: List) -> None:
        """Log translation request details."""
        logger.info(f"Translation request - Text: {len(text)} chars, Provider: {llm_provider}, "
                   f"Context: {use_context}, Chapter: {chapter_number}, Dict matches: {len(matches)}")

    def _log_translation_completion(self, text: str, translated: str, context_info: Optional[Dict],
                                   matches: List, chapter_number: Optional[int], use_context: bool) -> None:
        """Log translation completion details."""
        context_used = context_info is not None
        logger.info(f"Translation completed - Input: {len(text)} chars, Output: {len(translated)} chars, "
                   f"Context: {context_used}, Dict matches: {len(matches)}")

        if context_used:
            logger.info(f"Context-enhanced translation using Chapter {chapter_number}")
        elif use_context:
            logger.info(f"Context requested but not used for Chapter {chapter_number}")
        else:
            logger.info("Standard translation without context")
    
    def get_config(self) -> Dict[str, Union[List[str], Dict[str, str]]]:
        """Get service configuration including supported providers and defaults."""
        return {
            "supported_providers": [provider.value for provider in SupportedProvider],
            "default_models": {
                SupportedProvider.CHATGPT.value: "gpt-4o-mini",
                SupportedProvider.CHATGPT_AZURE.value: "gpt-4o-mini",
                SupportedProvider.DEEPSEEK.value: "deepseek-chat",
                SupportedProvider.QWEN.value: "qwen-turbo"
            },
            "default_tokens": {
                SupportedProvider.CHATGPT.value: os.getenv("OPENAI_API_KEY", ""),
                SupportedProvider.CHATGPT_AZURE.value: os.getenv("AZURE_OPENAI_API_KEY", ""),
                SupportedProvider.DEEPSEEK.value: os.getenv("DEEPSEEK_API_KEY", ""),
                SupportedProvider.QWEN.value: os.getenv("DASHSCOPE_API_KEY", "")
            }
        }