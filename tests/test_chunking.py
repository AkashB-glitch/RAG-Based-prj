import pytest

from app.ingestion.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunks_respect_max_size_and_cover_all_words():
    text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    seen = set(" ".join(chunks).split())
    assert seen == set(text.split())


def test_prefers_paragraph_breaks():
    para = "a" * 300
    chunks = chunk_text(f"{para}\n\n{para}\n\n{para}", max_chars=700, overlap=0)
    assert chunks[0] == f"{para}\n\n{para}"


def test_rejects_overlap_that_could_stall():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=100, overlap=60)
