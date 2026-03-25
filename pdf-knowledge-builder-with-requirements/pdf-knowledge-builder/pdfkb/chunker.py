from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=8000, chunk_overlap=1000):
    """
    Splits long text (multimodal) into overlapping chunks.
    This is required for:
      - FAISS vectorstore
      - Summarization
      - Research proposal generation
      - RAG pipelines

    Args:
        text (str): The raw multimodal text.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[str]: List of chunk strings.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # reduces fragmentation
    )

    chunks = splitter.split_text(text)
    return chunks
