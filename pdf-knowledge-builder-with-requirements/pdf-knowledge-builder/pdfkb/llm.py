from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm(model: str = "llama3", temperature: float = 0.2, max_tokens: int = 2048):
    """
    Returns a fully configured ChatOllama model.

    Args:
        model (str): Name of Ollama LLM installed on the system.
        temperature (float): Creativity level.
        max_tokens (int): Maximum output tokens.

    Returns:
        ChatOllama: Configured LLM instance.
    """
    try:
        return ChatOllama(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load Ollama model '{model}'. "
            f"Did you run: `ollama pull {model}`?\n\nOriginal error: {e}"
        )


def run_llm(prompt: str, model: str = "llama3", temperature: float = 0.2):
    """
    A simple helper for one-off prompt completion.
    Uses ChatPromptTemplate → LLM → OutputParser.

    Args:
        prompt (str): The user prompt.
        model (str): LLM to use.

    Returns:
        str: Model output text.
    """
    llm = get_llm(model=model, temperature=temperature)

    tpl = ChatPromptTemplate.from_template("{prompt}")
    chain = tpl | llm | StrOutputParser()

    return chain.invoke({"prompt": prompt})


def stream_llm(prompt: str, model: str = "llama3", temperature: float = 0.2):
    """
    Optional streaming generator for use in Streamlit.
    Returns a generator yielding chunks as they arrive.

    Example usage in Streamlit:
        for chunk in stream_llm("Explain LFP doping"):
            st.write(chunk)

    Args:
        prompt (str): Prompt content.
        model (str): LLM name.

    Returns:
        generator of str
    """

    llm = get_llm(model=model, temperature=temperature)

    tpl = ChatPromptTemplate.from_template("{prompt}")
    chain = tpl | llm | StrOutputParser()

    # ChatOllama supports .stream() for token streaming
    for chunk in chain.stream({"prompt": prompt}):
        yield chunk
