from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import get_llm

SUMMARY_TEMPLATE = """
You are generating a structured summary of a document.

Use ONLY the provided content to create a clear, pedagogical, well-organized summary

-----------------------------------
CONTENT TO ANALYZE:
{content}
"""

TEACHING_SUMMARY_TEMPLATE = """
You are generating a structured teaching chapter for a university-level course.

Use ONLY the provided content to create a clear, pedagogical, well-organized
chapter that can be included directly in a Jupyter Book.

The output must be in **Markdown**, and include the following sections,
each filled with high‑quality academic content:

# Title
Provide a concise, topic-specific title based on the content.

# Overview
Provide a 3–5 sentence high‑level explanation suitable for students.

# Key Concepts
List the most important ideas as bullet points.

# Detailed Explanation
Provide a clear, structured, pedagogical explanation of the material.
Break into subsections if appropriate.

# Important Figures and Diagrams (Text Descriptions)
Summaries of any images or figure-like information present.

# Tables (Interpretation)
Explain tables extracted from the PDF.

# Examples or Case Studies
Give 1–3 applied examples based on the content.

# Learning Objectives
Provide 4–6 bullet points describing what a student should learn.

# Summary
A concise recap of the chapter.

# Quiz Questions (no answers)
Provide 5–8 quiz questions that test conceptual understanding.

-----------------------------------
CONTENT TO ANALYZE:
{content}
"""


def summarize_for_teaching(chunks, model: str = "llama3", temperature: float = 0.2):
    """
    Create a complete Jupyter‑Book‑ready teaching chapter from PDF-derived text.

    Args:
        chunks (list[str]): List of RAG chunks from loader + chunker.
        model (str): Ollama LLM to use (e.g., "llama3", "mistral", "qwen2.5").
        temperature (float): LLM temperature.

    Returns:
        str: Markdown chapter text.
    """
    # Load LLM
    llm = get_llm(model=model, temperature=temperature)

    # Prepare prompt template
    prompt = PromptTemplate.from_template(TEACHING_SUMMARY_TEMPLATE)

    # Compose chain
    chain = prompt | llm | StrOutputParser()

    # Merge chunks into a single context block
    content = "\n\n".join(chunks)

    # Generate chapter
    md_chapter = chain.invoke({"content": content})

    return md_chapter

def summarize(chunks, model: str = "llama3", temperature: float = 0.2):
    """
    Summarize a PDF-derived text.

    Args:
        chunks (list[str]): List of RAG chunks from loader + chunker.
        model (str): Ollama LLM to use (e.g., "llama3", "mistral", "qwen2.5").
        temperature (float): LLM temperature.

    Returns:
        str: Markdown chapter text.
    """
    # Load LLM
    llm = get_llm(model=model, temperature=temperature)

    # Prepare prompt template
    prompt = PromptTemplate.from_template(SUMMARY_TEMPLATE)

    # Compose chain
    chain = prompt | llm | StrOutputParser()

    # Merge chunks into a single context block
    content = "\n\n".join(chunks)

    # Generate chapter
    md_chapter = chain.invoke({"content": content})

    return md_chapter
