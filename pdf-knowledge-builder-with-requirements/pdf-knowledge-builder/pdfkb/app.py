
import streamlit as st
import os

from pdfkb.loader import load_multimodal
from pdfkb.chunker import chunk_text
from pdfkb.embeddings import get_embeddings
from pdfkb.vectorstore import build_vectorstore, load_vectorstore
from pdfkb.summarizer import summarize_for_teaching
from pdfkb.proposer import generate_research_application
from pdfkb.bookbuilder import save_markdown, write_toc

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF Knowledge Builder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Knowledge Builder")
st.write("Create teaching materials, Jupyter Books, or research applications from PDFs — fully locally with Ollama.")

# ---------------------------------------------------------
# Sidebar: Upload PDFs & Select Model
# ---------------------------------------------------------
with st.sidebar:
    st.header("Upload & Settings")

    uploaded_pdfs = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.subheader("LLM Model")
    llm_model = st.text_input("Ollama model", value="llama3")

    st.subheader("Embedding Model")
    emb_model = st.text_input("Embedding model", value="nomic-embed-text")

    build_vectorstore_checkbox = st.checkbox(
        "Build FAISS Vectorstore (recommended for large documents)",
        value=True
    )

# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------
if uploaded_pdfs:
    st.success(f"{len(uploaded_pdfs)} PDF(s) uploaded.")

    if st.button("Process PDFs"):
        with st.spinner("Extracting text, tables, figures (OCR)..."):
            multimodal = load_multimodal(uploaded_pdfs)

        st.success("✅ Multimodal content extracted.")

        with st.spinner("Chunking text..."):
            chunks = chunk_text(multimodal)

        st.success("✅ Text chunked.")

        if build_vectorstore_checkbox:
            with st.spinner("Embedding & building vectorstore..."):
                emb = get_embeddings(emb_model)
                build_vectorstore(chunks, emb)
            st.success("✅ Vectorstore built & saved.")

        st.session_state["chunks"] = chunks
        st.success("✅ Ready for summarization or generation!")

# ---------------------------------------------------------
# Display Options: Summaries, Proposals, Jupyter Book
# ---------------------------------------------------------
if "chunks" in st.session_state:

    st.header("Generate Content")

    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        if st.button("📘 Summarize PDF"):
            with st.spinner("Summarizing entire PDF..."):
                summary = summarize_for_teaching(
                    st.session_state["chunks"],
                    model=llm_model
                )
            st.session_state["output"] = summary
            st.success("✅ PDF summarized!")

        if "output" in st.session_state:
            st.subheader("📄 Summary Output")
            st.markdown(st.session_state["output"])

    with col2:
        if st.button("📘 Generate Teaching Chapter"):
            with st.spinner("Generating teaching material..."):
                chapter = summarize_for_teaching(
                    st.session_state["chunks"],
                    model=llm_model
                )
            st.session_state["output"] = chapter
            st.success("✅ Teaching chapter created!")

    with col3:
        if st.button("📄 Generate Research Application"):
            with st.spinner("Generating research application..."):
                proposal = generate_research_application(
                    st.session_state["chunks"],
                    model=llm_model
                )
            st.session_state["output"] = proposal
            st.success("✅ Research application created!")

    with col4:
        if st.button("📚 Save as Jupyter Book Chapter"):
            text = st.session_state.get("output", "")
            if text.strip() == "":
                st.error("No content generated yet.")
            else:
                path = save_markdown(
                    text,
                    "my-jupyter-book/content",
                    "generated_chapter.md"
                )
                write_toc(["generated_chapter.md"], "my-jupyter-book")
                st.success(f"✅ Saved to {path}")

# ---------------------------------------------------------
# Output Preview
# ---------------------------------------------------------
if "output" in st.session_state:
    st.header("📄 Generated Output")
    st.markdown(st.session_state["output"])
