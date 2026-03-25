
# PDF Knowledge Builder

This project contains a modular pipeline for extracting multimodal content from PDFs, generating Jupyter Book chapters, and creating research applications using Ollama-based LLMs.

## Installation
- Install system dependencies: Tesseract OCR
- Install Python requirements: `pip install -r requirements.txt`
- Install Ollama models: `ollama pull llama3` and `ollama pull nomic-embed-text`

## Usage
Import the modules from `pdfkb/` to process PDFs, chunk text, build vectorstores, summarize content, or generate research applications.

Build a Jupyter Book using:
```
jupyter-book build my-jupyter-book/
```


## how to run:


pip install -r pdf-knowledge-builder/requirements.txt
sudo apt install tesseract-ocr     # Linux
ollama serve

ollama pull llama3
ollama pull nomic-embed-text

### jupyter book
python example_teaching_chapter.py

jupyter-book build my-jupyter-book/

### Generate a research application from PDFs: 

python example_research_application.py

### interactive use

jupyter notebook example_jupyternotebook.ipynb

### streamlit app 

streamlit run app.py
