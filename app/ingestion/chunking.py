import re

_SEPARATORS = ("\n\n", ". ", "\n", " ")


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph > sentence > line > word breaks.

    A break is only taken in the second half of the window, which guarantees forward progress
    as long as overlap < max_chars / 2.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars // 2:
        raise ValueError("require max_chars > 0 and 0 <= overlap < max_chars // 2")

    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx >= max_chars // 2:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if overlap:
            # don't start the next chunk mid-word: move forward to the next whitespace
            ws = re.search(r"\s", text[next_start:end])
            if ws:
                next_start += ws.end()
        start = next_start
    return chunks
