"""Automatic, catchy title generation for any video.

Given whatever signals are available (filename, description, content keywords,
or an optional topic hint), produce an engaging, well-formatted title. Works
fully offline.

Strategy:
  1. Work out a short "topic" phrase from the best available signal.
  2. Title-case it.
  3. Drop it into a catchy template chosen by style (stable per video).
  4. Optionally prepend a relevant emoji and trim to a max length.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, List, Optional

# Filename tokens that carry no meaning (camera/export junk).
_JUNK_TOKENS = {
    "vid", "video", "img", "image", "mov", "movie", "clip", "final",
    "export", "render", "output", "untitled", "new", "copy", "edit",
    "edited", "raw", "footage", "rec", "recording", "screen", "capture",
    "whatsapp", "download", "downloaded", "mp4", "mov", "reel", "short",
}

# Catchy templates. "{t}" is replaced with the title-cased topic.
_TEMPLATES = {
    "catchy": [
        "{t} You NEED to See!",
        "You Won't Believe This {t}",
        "{t} Like You've Never Seen Before",
        "This {t} Is INSANE",
        "Wait For It... {t}",
        "{t} That Broke The Internet",
        "Watch This {t} Till The End",
        "The {t} Everyone's Talking About",
    ],
    "question": [
        "Have You Ever Seen {t} Like This?",
        "What Happens When {t}?",
        "Is This The Best {t} Ever?",
        "Why Is Everyone Obsessed With {t}?",
        "Can You Guess What Happens In This {t}?",
    ],
    "clean": [
        "{t}",
        "{t} | Must Watch",
        "Amazing {t}",
        "{t} in 60 Seconds",
        "{t} — Quick Look",
    ],
}

# When there is no usable topic at all.
_GENERIC = {
    "catchy": [
        "You Won't Believe What Happens Next!",
        "This Will Make Your Day",
        "Wait For The End...",
        "Watch Till The End!",
        "This Is Too Good Not To Share",
    ],
    "question": [
        "Have You Seen Anything Like This?",
        "What Would You Do In This Moment?",
        "Can You Watch This Without Smiling?",
    ],
    "clean": [
        "Must Watch Video",
        "Today's Highlight",
        "Quick Clip You'll Love",
    ],
}

# Emoji hints by keyword (first match wins).
_EMOJI_MAP = [
    (("food", "recipe", "cooking", "eat", "snack", "kitchen"), "\U0001F374"),   # fork/knife
    (("workout", "fitness", "gym", "exercise", "yoga"), "\U0001F4AA"),           # flexed biceps
    (("coffee", "tea", "brew"), "\u2615"),                                       # hot beverage
    (("travel", "trip", "tour", "city", "beach"), "\u2708\uFE0F"),               # airplane
    (("game", "gaming", "gamer", "play"), "\U0001F3AE"),                          # video game
    (("money", "cash", "rich", "business", "earn"), "\U0001F4B0"),               # money bag
    (("tech", "phone", "gadget", "ai", "coding", "code"), "\U0001F4F1"),         # mobile phone
    (("music", "song", "dance", "beat"), "\U0001F3B5"),                          # musical note
    (("car", "bike", "drive", "race"), "\U0001F697"),                            # car
    (("dog", "cat", "pet", "puppy", "kitten", "animal"), "\U0001F436"),          # dog face
]
_DEFAULT_EMOJI = "\U0001F525"  # fire

# Connector words that look bad at the start/end of a topic phrase.
_CONNECTORS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "is", "are", "was", "were", "this", "that", "my",
    "your", "best", "how", "what", "when", "why",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-z0-9]+", text or "", re.IGNORECASE)]


def _trim_connectors(phrase: str) -> str:
    """Drop leading/trailing connector words so the topic reads cleanly."""
    words = phrase.split()
    while words and words[0].lower() in _CONNECTORS:
        words.pop(0)
    while words and words[-1].lower() in _CONNECTORS:
        words.pop()
    return " ".join(words)


def _is_clean_token(tok: str) -> bool:
    return bool(tok) and tok not in _JUNK_TOKENS and not tok.isdigit()


def _clean_keywords(keywords: Optional[Iterable[str]]) -> List[str]:
    """Keep only meaningful keywords (no junk words, no numbers)."""
    out: List[str] = []
    for k in keywords or []:
        if not k:
            continue
        parts = k.split()
        if all(_is_clean_token(p) for p in parts):
            out.append(k)
    return out


def _clean_filename_tokens(filename: Optional[str]) -> List[str]:
    if not filename:
        return []
    stem = Path(filename).stem
    tokens = _tokenize(re.sub(r"[._\-]+", " ", stem))
    # Drop junk words and pure numbers (timestamps, counters).
    return [t for t in tokens if t not in _JUNK_TOKENS and not t.isdigit()]


def _topic_from_description(description: str, max_words: int = 5) -> str:
    """Take the first meaningful chunk of the description as the topic."""
    if not description:
        return ""
    # First sentence / line.
    first = re.split(r"[.!?\n]", description.strip(), maxsplit=1)[0]
    words = first.split()
    return " ".join(words[:max_words]).strip()


def _pick_topic(
    description: str,
    keywords: Optional[Iterable[str]],
    filename: Optional[str],
    hint: Optional[str],
) -> str:
    """Choose the best available topic phrase."""
    if hint:
        return _trim_connectors(hint.strip())

    desc_topic = _trim_connectors(_topic_from_description(description))
    if desc_topic:
        return desc_topic

    # Prefer a multi-word keyword (bigram) if present, else top keywords.
    kw = _clean_keywords(keywords)
    bigrams = [k for k in kw if " " in k]
    if bigrams:
        return bigrams[0]
    if kw:
        return " ".join(kw[:3])

    file_tokens = _clean_filename_tokens(filename)
    if file_tokens:
        return " ".join(file_tokens[:3])

    return ""


def _title_case(text: str) -> str:
    small = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"}
    words = text.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i != 0 and lw in small:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _emoji_for(topic: str, keywords: Optional[Iterable[str]]) -> str:
    haystack = (topic + " " + " ".join(keywords or [])).lower()
    for needles, emoji in _EMOJI_MAP:
        if any(n in haystack for n in needles):
            return emoji
    return _DEFAULT_EMOJI


def generate_title(
    filename: Optional[str] = None,
    description: str = "",
    keywords: Optional[Iterable[str]] = None,
    hint: Optional[str] = None,
    style: str = "catchy",
    max_length: int = 70,
    emoji: bool = True,
    seed: Optional[str] = None,
) -> str:
    """Generate a catchy title for a video.

    ``style`` is one of: catchy, question, clean. ``max_length`` leaves room for
    hashtags that may be appended later (YouTube's hard title limit is 100).
    A ``seed`` (e.g. the filename) keeps the choice stable for the same video.
    """
    style = style if style in _TEMPLATES else "catchy"
    rng = random.Random(seed if seed is not None else filename)

    topic = _pick_topic(description, keywords, filename, hint)

    if topic:
        topic = _title_case(topic)
        template = rng.choice(_TEMPLATES[style])
        title = template.format(t=topic)
    else:
        title = rng.choice(_GENERIC[style])

    if emoji:
        title = f"{title} {_emoji_for(topic, keywords)}"

    title = re.sub(r"\s+", " ", title).strip()

    if len(title) > max_length:
        # Trim on a word boundary.
        trimmed = title[:max_length].rsplit(" ", 1)[0].rstrip(" -|")
        title = trimmed or title[:max_length]

    return title
