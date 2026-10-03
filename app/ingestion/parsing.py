from io import BytesIO
from pathlib import PurePath

EXT_TO_MIME = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf"}


def mime_from_filename(filename: str) -> str:
    """Decide type from the extension, not the client-supplied Content-Type header."""
    ext = PurePath(filename or "").suffix.lower()
    if ext not in EXT_TO_MIME:
        raise ValueError(f"unsupported file type {ext!r}; allowed: {sorted(EXT_TO_MIME)}")
    return EXT_TO_MIME[ext]


def extract_text(raw: bytes, mime_type: str) -> str:
    if mime_type in ("text/plain", "text/markdown"):
        return raw.decode("utf-8", errors="replace")
    if mime_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"unsupported mime type: {mime_type}")
