'''
Text Normalization - per language, two-tier
  normalize_basic — Tier 1 only
  normalize_full  — Tier 1 + Tier 2
  
Input: Raw transcript from dataset
Output: Cleaned normalized text strings
==============================================
'''

from abc import ABC
import unicodedata
import regex
import logging

logger = logging.getLogger(__name__)

# ==================================================================
# BASE
# ==================================================================
class BaseTextNormalizer(ABC):
    # Shared concrete primitives
    def _unicode_nfc(self, text):
        return unicodedata.normalize("NFC", text)
    
    def _lowercase(self, text):
        return text.lower()
    
    def _remove_punctuation(self, text):
        return regex.sub(r'[^\p{L}\p{N}\s]', ' ', text)
    
    def _collapse_whitespace(self, text):
        return regex.sub(r'\s+', ' ', text).strip()
    
    # Hook methods (subclasses override what they need)
    def _pre_tier1_hook(self, text):
        return text
    
    def _post_tier1_hook(self, text):
        return text
    
    def _tier2(self, text):
        return text
    
    # Shared pipelines
    def _tier1(self, text):
        # NFC, lowercase, pre_hook, remove_punct, collapse_ws, post_hook
        text = self._unicode_nfc(text)
        text = self._lowercase(text)
        text = self._pre_tier1_hook(text)
        text = self._remove_punctuation(text)
        text = self._collapse_whitespace(text)
        text = self._post_tier1_hook(text)
        return text
    
    def _full_pipeline(self, text):
        # NFC, lowercase, pre_hook, then tier2, remove_punct, collapse_ws, post_hook
        text = self._unicode_nfc(text)
        text = self._lowercase(text)
        text = self._pre_tier1_hook(text)
        text = self._tier2(text)
        text = self._remove_punctuation(text)
        text = self._collapse_whitespace(text)
        text = self._post_tier1_hook(text)
        return text 

    # Public API
    def normalize_basic(self, text):
        """Tier 1 only"""
        return self._tier1(text)
    
    def normalize_full(self, text):
        """Tier 1 + Tier 2 """
        return self._full_pipeline(text)

# ==================================================================
# VIETNAMESE
# ==================================================================
from vietnormalizer import VietnameseNormalizer

class VietnameseTextNormalizer(BaseTextNormalizer):

    def __init__(self):
        self._vn = VietnameseNormalizer()

    def _tier2(self, text):
        """Apply VietNormalizer expansion (numbers, dates, currency, acronyms,
        and foreign-word transliteration)."""
        try:
            return self._vn.normalize(text, enable_transliteration=True)
        except Exception as e:
            logger.warning("VietNormalizer failed on input %r: %s", text[:80], e)
            return text

# ==================================================================
# MANDARIN
# ==================================================================
import cn2an

class MandarinTextNormalizer(BaseTextNormalizer):
    def _post_tier1_hook(self, text):
        text = regex.sub(r'\s+', '', text) # strip spaces, run last
        return text

    def _tier2(self, text):
        try:
            return cn2an.transform(text, "an2cn")
        except Exception as e:
            logger.warning("MandarinTextNormalizer failed on input %r: %s", text[:80], e)
            return text
       
# ==================================================================
# FRENCH
# ==================================================================
from num2words import num2words

class FrenchTextNormalizer(BaseTextNormalizer):
    _TYPOGRAPHIC_APOS_TRANS = str.maketrans({"\u2019": "'", "\u02BC": "'", "\u2018": "'"})
    
    def _remove_punctuation(self, text):
        return regex.sub(r"[^\p{L}\p{N}\s']", ' ', text)

    def _normalize_apostrophe(self, text):
        text = text.translate(self._TYPOGRAPHIC_APOS_TRANS)
        text = regex.sub(r"(?<=\p{L})\s+'\s+(?=\p{L})", "'", text) # Handle apostrophe cases
        return text
    
    def _expand_digits(self, text):
        return regex.sub(
            r"\d+",
            lambda m: num2words(int(m.group()), lang="fr"),
            text,
            )
        
    def _pre_tier1_hook(self, text):
        text = self._normalize_apostrophe(text)
        text = text.replace("-", " ")
        return text
    
    def _tier2(self, text):
        try:
            return self._expand_digits(text)
        except Exception as e:
            logger.warning("FrenchTextNormalizer failed on input %r: %s", text[:80], e)
            return text

# ==================================================================
# ASTURIAN 
# ==================================================================
class AsturianTextNormalizer(BaseTextNormalizer):
    _TYPOGRAPHIC_APOS_TRANS = str.maketrans({"\u2019": "'", "\u02BC": "'", "\u2018": "'"})
    _NUM_RE = regex.compile(r"\d{1,3}(?:[.\u00a0 ]\d{3})+|\d+")

    def _remove_punctuation(self, text):
        return regex.sub(r"[^\p{L}\p{N}\s']", ' ', text)

    def _normalize_apostrophe(self, text):
        text = text.translate(self._TYPOGRAPHIC_APOS_TRANS)
        text = regex.sub(r"(?<=\p{L})\s+'\s+(?=\p{L})", "'", text)
        return text

    def _expand_digits(self, text):
        def _sub(m):
            digits = regex.sub(r"[.\u00a0 ]", "", m.group())
            return num2words(int(digits), lang="es")
        return self._NUM_RE.sub(_sub, text)

    def _pre_tier1_hook(self, text):
        text = self._normalize_apostrophe(text)
        text = text.replace("-", " ")
        return text

    def _tier2(self, text):
        try:
            return self._expand_digits(text)
        except Exception as e:
            logger.warning("AsturianTextNormalizer failed on input %r: %s", text[:80], e)
            return text

# ==================================================================
# SPANISH 
# ==================================================================
class SpanishTextNormalizer(BaseTextNormalizer):
    _TYPOGRAPHIC_APOS_TRANS = str.maketrans({"\u2019": "'", "\u02BC": "'", "\u2018": "'"})
    _NUM_RE = regex.compile(r"\d{1,3}(?:[.\u00a0 ]\d{3})+|\d+")

    def _remove_punctuation(self, text):
        return regex.sub(r"[^\p{L}\p{N}\s']|[ºª]", ' ', text)

    def _normalize_apostrophe(self, text):
        text = text.translate(self._TYPOGRAPHIC_APOS_TRANS)
        text = regex.sub(r"(?<=\p{L})\s+'\s+(?=\p{L})", "'", text)
        return text

    def _expand_digits(self, text):
        def _sub(m):
            digits = regex.sub(r"[.\u00a0 ]", "", m.group())   
            return num2words(int(digits), lang="es")
        return self._NUM_RE.sub(_sub, text)

    def _pre_tier1_hook(self, text):
        text = self._normalize_apostrophe(text)
        text = text.replace("-", " ")
        return text

    def _tier2(self, text):
        try:
            return self._expand_digits(text)
        except Exception as e:
            logger.warning("SpanishTextNormalizer failed on input %r: %s", text[:80], e)
            return text

# ==================================================================
# FACTORY
# ==================================================================
_REGISTRY = {
    "vi_vn":       VietnameseTextNormalizer,
    "cmn_hans_cn": MandarinTextNormalizer,
    "fr_fr":       FrenchTextNormalizer,
    "ast_es":      AsturianTextNormalizer,
    "es_419":      SpanishTextNormalizer,
}

def get_normalizer(language: str) -> BaseTextNormalizer:
    key = language.lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No normalizer registered for language={language!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return cls()