from langchain_community.embeddings import OllamaEmbeddings

def get_embeddings(model: str = "nomic-embed-text"):
    """
    Load and return an Ollama embedding model.

    Args:
        model (str): Name of the Ollama embedding model to use.
                     Recommended defaults:
                        - "nomic-embed-text"
                        - "mistral"
                        - "llama3"

    Returns:
        OllamaEmbeddings: embedding model instance
    """

    try:
        emb = OllamaEmbeddings(model=model)
        return emb
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model}'. "
            f"Make sure it is installed via: `ollama pull {model}`\n\n"
            f"Original error: {e}"
        )


def embed_text_list(text_list, model: str = "nomic-embed-text"):
    """
    Helper function: embed a list of strings manually.

    Args:
        text_list (list[str]): List of text chunks.
        model (str): Embedding model name.

    Returns:
        list[list[float]]: List of embeddings (vectors)
    """
    emb = get_embeddings(model)
    return emb.embed_documents(text_list)


def embed_single(text: str, model: str = "nomic-embed-text"):
    """
    Helper function: embed a single text string.

    Args:
        text (str): The text to embed.
        model (str): Embedding model name.

    Returns:
        list[float]: Vector embedding
    """
    emb = get_embeddings(model)
    return emb.embed_query(text)
