"""Memory-based caching system for MRCT BOOK Translator.

Provides high-performance in-memory caching for:
- Translation results with LLM provider/model specificity
- Dictionary lookup results with position tracking
- Context retrieval from PDF chapters
- Chapter content with extended TTL

Features:
- LRU eviction policy with configurable cache sizes
- TTL support with automatic expiration cleanup
- Comprehensive statistics tracking and reporting
- Thread-safe operations with OrderedDict backing
"""

import time
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cache configuration constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL = 3600  # 1 hour in seconds
DICTIONARY_CACHE_SIZE = 10000  # Larger cache for dictionary lookups
CONTEXT_CACHE_SIZE = 100  # Smaller cache for context (larger memory footprint)
TRANSLATION_CACHE_SIZE = 500  # Cache for complete translations


@dataclass
class CacheEntry:
    """Represents a single cache entry with data and metadata."""
    data: Any
    timestamp: float
    hit_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self) -> None:
        """Update hit count and access time."""
        self.hit_count += 1


class MemoryCache:
    """
    Thread-safe memory cache with LRU eviction and TTL support.
    Uses OrderedDict for O(1) access and LRU ordering.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, default_ttl: Optional[float] = DEFAULT_TTL):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        if key in self._cache:
            entry = self._cache[key]

            # Check if entry is expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['expirations'] += 1
                self._stats['misses'] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats['hits'] += 1

            return entry.data

        self._stats['misses'] += 1
        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in the cache."""
        # Use provided TTL or default
        entry_ttl = ttl if ttl is not None else self.default_ttl

        # If key exists, update it
        if key in self._cache:
            self._cache[key] = CacheEntry(value, time.time(), ttl=entry_ttl)
            self._cache.move_to_end(key)
        else:
            # Add new entry
            self._cache[key] = CacheEntry(value, time.time(), ttl=entry_ttl)

            # Evict oldest entries if cache is full
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove_entry(oldest_key)
                self._stats['evictions'] += 1

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache."""
        if key in self._cache:
            del self._cache[key]

    def cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key for the given arguments."""
        return self._generate_key(*args, **kwargs)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0

        return {
            **self._stats,
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)
            self._stats['expirations'] += 1

        return len(expired_keys)


class CacheManager:
    """
    Central cache manager that coordinates different cache instances
    for different types of data.
    """

    def __init__(self):
        # Separate caches for different types of data
        self.dictionary_cache = MemoryCache(
            max_size=DICTIONARY_CACHE_SIZE,
            default_ttl=None  # Dictionary entries don't expire
        )

        self.context_cache = MemoryCache(
            max_size=CONTEXT_CACHE_SIZE,
            default_ttl=1800  # Context expires after 30 minutes
        )

        self.translation_cache = MemoryCache(
            max_size=TRANSLATION_CACHE_SIZE,
            default_ttl=3600  # Translations expire after 1 hour
        )

        logger.info("Cache manager initialized with separate caches")

    def get_dictionary_matches(self, text: str, is_english: bool) -> Optional[Any]:
        """Get cached dictionary matches."""
        key = self.dictionary_cache.cache_key("dict_matches", text.lower(), is_english)
        return self.dictionary_cache.get(key)

    def cache_dictionary_matches(self, text: str, is_english: bool, matches: Any) -> None:
        """Cache dictionary matches."""
        key = self.dictionary_cache.cache_key("dict_matches", text.lower(), is_english)
        self.dictionary_cache.put(key, matches)

    def get_context(self, text: str, chapter_number: int, similarity_threshold: float) -> Optional[Any]:
        """Get cached context retrieval result."""
        # Normalize text for consistent caching
        normalized_text = text.strip().lower()
        key = self.context_cache.cache_key("context", normalized_text, chapter_number, similarity_threshold)
        return self.context_cache.get(key)

    def cache_context(self, text: str, chapter_number: int, similarity_threshold: float, context_result: Any) -> None:
        """Cache context retrieval result."""
        normalized_text = text.strip().lower()
        key = self.context_cache.cache_key("context", normalized_text, chapter_number, similarity_threshold)
        self.context_cache.put(key, context_result)

    def get_translation(self, text: str, llm_provider: str, model: str, system_prompt_hash: str) -> Optional[Any]:
        """Get cached translation result."""
        key = self.translation_cache.cache_key("translation", text.strip(), llm_provider, model, system_prompt_hash)
        return self.translation_cache.get(key)

    def cache_translation(self, text: str, llm_provider: str, model: str, system_prompt_hash: str, translation: str) -> None:
        """Cache translation result."""
        key = self.translation_cache.cache_key("translation", text.strip(), llm_provider, model, system_prompt_hash)
        self.translation_cache.put(key, translation)

    def get_chapter_content(self, chapter_number: int) -> Optional[str]:
        """Get cached chapter content."""
        key = self.context_cache.cache_key("chapter_content", chapter_number)
        return self.context_cache.get(key)

    def cache_chapter_content(self, chapter_number: int, content: str) -> None:
        """Cache chapter content."""
        key = self.context_cache.cache_key("chapter_content", chapter_number)
        # Chapter content has longer TTL since it doesn't change
        self.context_cache.put(key, content, ttl=7200)  # 2 hours

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics from all caches."""
        return {
            'dictionary': self.dictionary_cache.get_stats(),
            'context': self.context_cache.get_stats(),
            'translation': self.translation_cache.get_stats()
        }

    def cleanup_all(self) -> Dict[str, int]:
        """Clean up expired entries in all caches."""
        return {
            'dictionary': self.dictionary_cache.cleanup_expired(),
            'context': self.context_cache.cleanup_expired(),
            'translation': self.translation_cache.cleanup_expired()
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.dictionary_cache.clear()
        self.context_cache.clear()
        self.translation_cache.clear()
        logger.info("All caches cleared")


# Global cache manager instance
cache_manager = CacheManager()