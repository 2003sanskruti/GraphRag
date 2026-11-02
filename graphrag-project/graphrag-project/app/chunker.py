"""Step 2: split text into overlapping chunks on sentence boundaries."""
import re
from typing import Dict, List, Tuple

from .config import CHUNK_OVERLAP, CHUNK_SIZE

_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_pages(pages: List[Tuple[int, str]], size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    chunks: List[Dict] = []
    for page_no, text in pages:
        text = re.sub(r"[ \t]+", " ", text)
        sentences = [s.strip() for s in _SENT.split(text) if s and s.strip()]
        buf: List[str] = []
        length = 0
        for sent in sentences:
            # very long sentence: hard split
            while len(sent) > size:
                if buf:
                    chunks.append({"text": " ".join(buf), "page": page_no})
                    buf, length = [], 0
                chunks.append({"text": sent[:size], "page": page_no})
                sent = sent[size - overlap:]
            if length + len(sent) > size and buf:
                chunks.append({"text": " ".join(buf), "page": page_no})
                tail, tl = [], 0
                for s in reversed(buf):
                    if tl + len(s) > overlap:
                        break
                    tail.insert(0, s)
                    tl += len(s)
                buf, length = tail, tl
            buf.append(sent)
            length += len(sent) + 1
        if buf:
            chunks.append({"text": " ".join(buf), "page": page_no})
    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks
