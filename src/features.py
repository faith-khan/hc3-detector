from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

_SENT_RE = re.compile(r"[.!?]+")  # fallback splitter if punkt fails
_WORD_RE = re.compile(r"[A-Za-z]+") # drop digits, punctuation etc. numbers/emojis wont count toward length.

_PRONOUNS = {"i","me","my","mine","myself",
             "we","us","our","ours","ourselves",
             "you","your","yours","yourself","yourselves"}

def _sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: simple punctuation-based split
        return [s for s in _SENT_RE.split(text) if s.strip()]

def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]

def avg_sentence_length(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    lengths = [len(_words(s)) for s in sents]
    return float(np.mean(lengths)) if lengths else 0.0

# type token ratio: unique words / total words for vocab richness
#* may use length-aware variants of this in the future
def type_token_ratio(text: str) -> float:
    toks = _words(text)
    n = len(toks)
    return (len(set(toks)) / n) if n else 0.0

def pronoun_rate(text: str) -> float:
    toks = _words(text)
    n = len(toks)
    if n == 0:
        return 0.0
    return sum(t in _PRONOUNS for t in toks) / n

class StylisticFeatures(BaseEstimator, TransformerMixin):
    """Compute simple, interpretable features from raw text."""
    feature_names_: list[str] = ["avg_sentence_length", "type_token_ratio", "pronoun_rate"]

    def fit(self, X: Iterable[str], y=None):
        return self  # nothing to learn

    def transform(self, X: Iterable[str]) -> np.ndarray:
        vals = []
        for t in X:
            vals.append([
                avg_sentence_length(t),
                type_token_ratio(t),
                pronoun_rate(t)
            ])
        return np.asarray(vals, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
