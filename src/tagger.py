"""Automatic tag generation.

Generates relevant tags from a video's title, description and file name using
lightweight keyword extraction: tokenise -> drop stopwords -> score by
frequency + position + bigrams -> return the top N tags.

No external NLP dependency required, so it runs anywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

# A compact English stopword list (enough for tag extraction).
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "our", "your", "their", "my", "me", "us", "do", "does", "did", "done",
    "have", "has", "had", "will", "would", "can", "could", "should", "shall",
    "may", "might", "must", "not", "no", "yes", "so", "up", "down", "out",
    "about", "into", "over", "after", "before", "again", "more", "most",
    "very", "just", "how", "what", "when", "where", "who", "why", "which",
    "all", "any", "some", "such", "than", "too", "also", "here", "there",
    "video", "watch", "new", "official", "vid", "clip", "clips", "footage",
    "reel", "reels", "short", "shorts", "img", "mov", "mp4", "rec", "raw",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)

# Default USA-targeted hashtags. Used to reach a US audience and ride
# location-based discovery/trends. Override via config (tags.region_hashtags).
USA_HASHTAGS: List[str] = [
    "usa",
    "america",
    "usa_tiktok",
    "trendingusa",
    "viralusa",
    "unitedstates",
    "fyp",
    "fypusa",
    "americanlife",
    "madeinusa",
]


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def to_hashtag(text: str) -> str:
    """Convert a tag/phrase into a single #hashtag (alphanumeric only)."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", text)
    return f"#{cleaned}" if cleaned else ""


def build_title_with_hashtags(
    title: str,
    tags: Iterable[str],
    region_hashtags: Optional[Iterable[str]] = None,
    count: int = 3,
    max_length: int = 100,
) -> str:
    """Append hashtags to a title without exceeding ``max_length``.

    Picks the top content ``tags`` plus the first region hashtag, converts them
    to ``#hashtag`` form, de-duplicates against words already in the title, and
    appends as many as fit (YouTube title hard limit is 100 chars).
    """
    base = (title or "").strip()
    existing = {w.lower().lstrip("#") for w in base.split()}

    candidates: List[str] = []
    # Region hashtag first so US targeting is prioritised in the title.
    region = list(region_hashtags or [])
    if region:
        candidates.append(region[0])
    candidates.extend(tags)

    chosen: List[str] = []
    seen = set(existing)
    for cand in candidates:
        if len(chosen) >= count:
            break
        norm = re.sub(r"[^0-9a-zA-Z]+", "", cand).lower()
        if not norm or norm in seen:
            continue
        chosen.append(to_hashtag(cand))
        seen.add(norm)

    result = base
    for tag in chosen:
        candidate = f"{result} {tag}" if result else tag
        if len(candidate) <= max_length:
            result = candidate
        else:
            break
    return result


def add_region_hashtags(
    tags: List[str],
    region_hashtags: Iterable[str],
    max_tags: int,
) -> List[str]:
    """Merge region hashtags into ``tags`` (region tags first), de-duplicated.

    Region hashtags are stored as plain words here (e.g. ``usa``); the platform
    layer decides where a ``#`` is needed.
    """
    result: List[str] = []
    seen = set()
    for tag in list(region_hashtags) + list(tags):
        norm = tag.strip().lower()
        if norm and norm not in seen:
            result.append(norm)
            seen.add(norm)
        if len(result) >= max_tags:
            break
    return result


def _clean_filename(filename: Optional[str]) -> str:
    if not filename:
        return ""
    stem = Path(filename).stem
    # Replace common separators with spaces.
    return re.sub(r"[._\-]+", " ", stem)


def generate_tags(
    title: str = "",
    description: str = "",
    filename: Optional[str] = None,
    max_tags: int = 15,
    always_include: Optional[Iterable[str]] = None,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return a de-duplicated list of up to ``max_tags`` tags.

    ``always_include`` tags are placed first and always kept. The remaining
    slots are filled with the highest-scoring keywords and bigrams.
    """
    stopwords = set(_STOPWORDS)
    if extra_stopwords:
        stopwords |= {w.lower() for w in extra_stopwords}

    # Weight title highest, then filename, then description.
    title_tokens = _tokenize(title)
    file_tokens = _tokenize(_clean_filename(filename))
    desc_tokens = _tokenize(description)

    scores: Counter = Counter()

    def add_tokens(tokens: List[str], weight: float) -> None:
        for idx, tok in enumerate(tokens):
            if tok in stopwords or len(tok) < 3 or tok.isdigit():
                continue
            # Earlier tokens score slightly higher.
            position_bonus = max(0.0, 1.0 - idx * 0.02)
            scores[tok] += weight + position_bonus

    add_tokens(title_tokens, weight=3.0)
    add_tokens(file_tokens, weight=2.0)
    add_tokens(desc_tokens, weight=1.0)

    # Add meaningful bigrams from the title (good multi-word tags).
    def add_bigrams(tokens: List[str], weight: float) -> None:
        for a, b in zip(tokens, tokens[1:]):
            if a in stopwords or b in stopwords:
                continue
            if len(a) < 3 or len(b) < 3 or a.isdigit() or b.isdigit():
                continue
            scores[f"{a} {b}"] += weight

    add_bigrams(title_tokens, weight=2.5)

    # Build ordered result.
    result: List[str] = []
    seen = set()

    for tag in (always_include or []):
        norm = tag.strip().lower()
        if norm and norm not in seen:
            result.append(norm)
            seen.add(norm)

    for tag, _score in scores.most_common():
        if len(result) >= max_tags:
            break
        if tag not in seen:
            result.append(tag)
            seen.add(tag)

    return result[:max_tags]
