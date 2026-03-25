import os
import streamlit as st

from pdfkb.loader import load_multimodal
from pdfkb.chunker import chunk_text
from pdfkb.summarizer import summarize_for_teaching
from pdfkb.proposer import generate_research_application
from pdfkb.bookbuilder import save_markdown, write_toc



import subprocess, json

def scan_ollama_models():
    """
    Reads installed models directly from Ollama using:
        ollama list --json
    This works on Linux, macOS, Windows, system-service, user-mode, and custom paths.
    """
    try:
        result = subprocess.run(
            ["ollama", "list", "--json"],
            capture_output=True,
            text=True
        )

        # Parse JSON output
        items = json.loads(result.stdout)

        # Extract model names
        return [m["name"] for m in items]

    except Exception as e:
        print("Model scan failed:", e)
        return ["llama3", "mistral", "qwen2.5"]  # safe fallback



# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF Knowledge Builder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Knowledge Builder")
st.write("Turn PDFs into teaching material, summaries, and research applications.")

# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
  
with st.sidebar:
    st.header("Settings")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.subheader("Select LLM Model")

    available_models = scan_ollama_models()
    llm_model = st.selectbox(
        "Available Ollama models:",
        available_models,
        index=0
    )

    st.subheader("Embedding Model")
    emb_model = st.selectbox(
        "Embedding model:",
        ["nomic-embed-text", "mistral", "llama3"],
        index=0
    )

    if st.button("Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------------------------------------------
# If PDFs uploaded
# ---------------------------------------------------------
if uploaded_pdfs:
    st.success(f"{len(uploaded_pdfs)} PDF(s) uploaded.")

    # PROCESS PDFS BUTTON
    if st.button("Process PDFs"):
        with st.spinner("Extracting text, tables, figures (OCR)..."):
            multimodal = load_multimodal(uploaded_pdfs)

        with st.spinner("Chunking..."):
            chunks = chunk_text(multimodal)

        st.session_state["multimodal"] = multimodal
        st.session_state["chunks"] = chunks

        st.success("✅ PDFs processed!")

# ---------------------------------------------------------
# ACTION BUTTONS (always visible when PDFs are uploaded)
# ---------------------------------------------------------
if uploaded_pdfs:

    st.header("Actions")

    col1, col2, col3 = st.columns(3)

    # ✅ SUMMARIZE PDF
    with col1:
        if st.button("📘 Summarize PDF"):
            if "chunks" not in st.session_state:
                st.warning("Please click 'Process PDFs' first.")
            else:
                with st.spinner("Creating summary..."):
                    summary = summarize_for_teaching(
                        st.session_state["chunks"],
                        model=llm_model
                    )
                st.session_state["output"] = summary
                st.success("✅ Summary created!")

    # ✅ TEACHING CHAPTER
    with col2:
        if st.button("📗 Generate Teaching Chapter"):
            if "chunks" not in st.session_state:
                st.warning("Please click 'Process PDFs' first.")
            else:
                with st.spinner("Generating chapter..."):
                    chapter = summarize_for_teaching(
                        st.session_state["chunks"],
                        model=llm_model
                    )
                st.session_state["output"] = chapter
                st.success("✅ Teaching chapter created!")

    # ✅ RESEARCH APPLICATION
    with col3:
        if st.button("📄 Generate Research Application"):
            if "chunks" not in st.session_state:
                st.warning("Please click 'Process PDFs' first.")
            else:
                with st.spinner("Generating proposal..."):
                    app_text = generate_research_application(
                        st.session_state["chunks"],
                        model=llm_model
                    )
                st.session_state["output"] = app_text
                st.success("✅ Research application created!")

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
if "output" in st.session_state:
    st.header("📄 Output")
    st.markdown(st.session_state["output"])

    # SAVE TO JUPYTER BOOK
    if st.button("💾 Save as Jupyter Book Chapter"):
        filename = "generated_chapter.md"
        path = save_markdown(
            st.session_state["output"],
            "my-jupyter-book/content",
            filename
        )
        write_toc([filename], "my-jupyter-book")
        st.success(f"✅ Saved to {path}")
