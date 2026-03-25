from pdfkb.loader import load_multimodal
from pdfkb.chunker import chunk_text
from pdfkb.embeddings import get_embeddings
from pdfkb.vectorstore import build_vectorstore
from pdfkb.summarizer import summarize_for_teaching
from pdfkb.bookbuilder import save_markdown, write_toc

# 1. List your input PDFs
pdfs = [
    "pdfs/file1.pdf",
    "pdfs/file2.pdf",
]

# 2. Load multimodal content (text + tables + OCR figures)
multimodal_text = load_multimodal(pdfs)

# 3. Chunk the data for processing
chunks = chunk_text(multimodal_text)

# 4. Build vectorstore (optional if only summarizing)
emb = get_embeddings("nomic-embed-text")
build_vectorstore(chunks, emb)

# 5. Generate a teaching-ready chapter
chapter_md = summarize_for_teaching(chunks)

# 6. Save to Jupyter Book folder
filename = "chapter1.md"
save_markdown(chapter_md, "my-jupyter-book/content", filename)

# 7. Update table of contents
write_toc([filename], "my-jupyter-book")

print("✅ Teaching chapter created!")
