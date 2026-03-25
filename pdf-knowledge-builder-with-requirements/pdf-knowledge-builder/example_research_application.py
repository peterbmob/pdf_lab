from pdfkb.loader import load_multimodal
from pdfkb.chunker import chunk_text
from pdfkb.proposer import generate_research_application

pdfs = [
    "pdfs/background1.pdf",
    "pdfs/literature_review.pdf",
]

# Load multimodal PDF content
text = load_multimodal(pdfs)

# Chunk for LLM input
chunks = chunk_text(text)

# Generate a research application
proposal = generate_research_application(chunks, model="llama3")

# Save
with open("research_application.md", "w", encoding="utf-8") as f:
    f.write(proposal)

print("✅ Research application created in research_application.md!")