"""Split long text at sentence boundaries so each model call stays within what the model handles well."""

import re

_SENTENCE_RE = re.compile(r"[^.!?…。！？]+(?:[.!?…。！？]+[\"'”’)\]]*|$)")
_CLAUSE_RE = re.compile(r"[^,;:—–]+(?:[,;:—–]+|$)")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """A sentence longer than max_chars is cut at clause punctuation, then at spaces."""
    parts: list[str] = []
    current = ""
    pieces = [c.strip() for c in _CLAUSE_RE.findall(sentence) if c.strip()]
    for piece in pieces:
        if len(piece) > max_chars:
            words = piece.split()
            for w in words:
                if current and len(current) + 1 + len(w) > max_chars:
                    parts.append(current)
                    current = w
                else:
                    current = f"{current} {w}".strip()
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Group whole sentences into chunks of at most max_chars characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        pieces = _split_long(sentence, max_chars) if len(sentence) > max_chars else [sentence]
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks
