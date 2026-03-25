import pdfplumber
from pypdf import PdfReader
import pytesseract
from PIL import Image


def extract_text(pdf_files):
    """Extract plain text from all PDF pages."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_tables(pdf_files):
    """Extract tables and convert them into text."""
    table_text = ""
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_str = " | ".join(cell if cell else "" for cell in row)
                        table_text += row_str + "\n"
    return table_text


def extract_figures(pdf_files):
    """OCR figures/images embedded in PDFs."""
    ocr_text = ""
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                for img in page.images:
                    try:
                        # Crop bounding box
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        cropped = page.crop(bbox).to_image(resolution=300).original
                        ocr_text += pytesseract.image_to_string(cropped) + "\n"
                    except Exception:
                        pass
    return ocr_text


def load_multimodal(pdf_files):
    """
    Returns text + tables + OCR figures as a SINGLE combined string.
    This is the main function expected by the Streamlit app.
    """
    text = extract_text(pdf_files)
    tables = extract_tables(pdf_files)
    figures = extract_figures(pdf_files)

    combined = (
        "## TEXT\n" + text +
        "\n\n## TABLES\n" + tables +
        "\n\n## FIGURES (OCR)\n" + figures
    )

    return combined
``
