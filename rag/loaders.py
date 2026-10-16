from pathlib import Path


def load_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_document(path: str) -> str:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        text = load_pdf(path)
    elif suffix == ".txt":
        text = load_txt(path)
    else:
        raise ValueError(
            f"Unsupported document type '{suffix}' for file: {path}. "
            f"Supported types: .pdf, .txt"
        )
    if not text.strip():
        raise ValueError(f"No extractable text found in file: {path}")
    return text
