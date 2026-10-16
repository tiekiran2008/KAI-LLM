def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        char_count = 0
        end = start
        while end < len(words):
            word_len = len(words[end]) + (1 if end > start else 0)
            if char_count + word_len > chunk_size and end > start:
                break
            char_count += word_len
            end += 1

        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        keep_chars = max(0, char_count - chunk_overlap)
        consumed = 0
        advance = start
        while advance < end and consumed < keep_chars:
            consumed += len(words[advance]) + 1
            advance += 1
        start = max(start + 1, advance)

    return chunks
