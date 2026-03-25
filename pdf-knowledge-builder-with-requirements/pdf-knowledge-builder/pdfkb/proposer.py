from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import get_llm


RESEARCH_PROPOSAL_TEMPLATE = """
You are assisting in writing a **professional research application** suitable for
major scientific funding bodies (e.g., European Commission, VR, SSF, ERC).

Using ONLY the content provided, generate a fully structured research proposal.

Your output MUST be in Markdown and include the following sections:

# Project Title

# Executive Summary (150–250 words)

# Background and Scientific Context
- What is known?
- Why the topic is important?
- Summary of key findings from the provided material.

# State-of-the-Art and Related Work
- What has been done previously?
- What gaps remain?

# Identified Knowledge Gaps
- Clear, bullet-pointed research gaps derived from the content.

# Research Questions / Hypotheses
- 3–6 sharp, testable questions or hypotheses.

# Project Objectives
- High-level objectives
- Measurable, concrete goals

# Methodology and Work Plan
- Methods, techniques, tools
- Experimental or computational workflows
- Data sources
- Justify why these methods are appropriate

# Work Packages (WP1, WP2, WP3…)
For each WP:
- Title
- Objectives
- Tasks
- Deliverables

# Expected Results
- Scientific outcomes
- Technical outputs
- Educational or societal impact

# Impact and Relevance
- Scientific impact
- Societal or industrial relevance
- Why the proposed work matters now

# Budget Narrative (text only)
- How resources will be used
- Personnel, equipment, facility access

# Risk Analysis and Mitigation
- Scientific risks
- Technical risks
- Mitigation strategies

# References (based on the input material)
- Summaries only (do not fabricate)

---------------------------------------------

CONTENT TO ANALYZE:
{content}
"""


def generate_research_application(chunks, model: str = "llama3", temperature: float = 0.2):
    """
    Generates a full research application using multimodal PDF-derived chunks.

    Args:
        chunks (list[str]): Text chunks from loader + chunker.
        model (str): Ollama model name.
        temperature (float): Creativity level.

    Returns:
        str: Markdown research application.
    """

    llm = get_llm(model=model, temperature=temperature)

    prompt = PromptTemplate.from_template(RESEARCH_PROPOSAL_TEMPLATE)

    chain = prompt | llm | StrOutputParser()

    content = "\n\n".join(chunks)

    return chain.invoke({"content": content})
