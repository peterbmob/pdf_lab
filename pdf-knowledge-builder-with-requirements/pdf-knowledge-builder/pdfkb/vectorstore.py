import os
from langchain_community.vectorstores import FAISS


def build_vectorstore(chunks, embeddings, index_path: str = "faiss_index"):
    """
    Build a FAISS vectorstore from list of text chunks and save it locally.

    Args:
        chunks (list[str]): The text chunks to embed and store.
        embeddings: An OllamaEmbeddings instance from get_embeddings().
        index_path (str): Directory to save the FAISS index.
    """
    if not chunks:
        raise ValueError("No chunks provided for building vectorstore.")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Save local store
    vectorstore.save_local(index_path)
    return index_path


def load_vectorstore(embeddings, index_path: str = "faiss_index"):
    """
    Load a FAISS vectorstore from local disk.

    Args:
        embeddings: Embedding model used originally
        index_path (str): Directory containing FAISS index files.

    Returns:
        FAISS instance or raises error if index missing.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            f"Please build the index first."
        )

    store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return store


def reset_vectorstore(index_path: str = "faiss_index"):
    """
    Delete a FAISS index from disk.
    Useful when switching PDFs or embeddings.
    """
    if os.path.exists(index_path):
        for f in os.listdir(index_path):
            os.remove(os.path.join(index_path, f))
        os.rmdir(index_path)
        return True
    return False


def similarity_search(query: str, store: FAISS, k: int = 4):
    """
    Run similarity search on an already loaded FAISS store.

    Args:
        query (str): User query
        store (FAISS): Loaded vectorstore
        k (int): Number of documents to return

    Returns:
        list[Document]: Top-k relevant documents
    """
    return store.similarity_search(query, k=k)
