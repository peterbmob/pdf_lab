import os
from datetime import datetime

def ensure_dir(path: str):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    """
    Converts any title into a safe filename.
    Example: "Chapter 1: Intro" → "chapter_1_intro.md"
    """
    name = name.lower()
    bad_chars = " :/\\|?*<>'\""
    for c in bad_chars:
        name = name.replace(c, "_")
    return name + ".md" if not name.endswith(".md") else name


def save_markdown(text: str, out_dir: str, filename: str) -> str:
    """
    Save generated Markdown content to a target folder.
    Returns the full path to the saved file.
    """
    ensure_dir(out_dir)

    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


def write_toc(chapter_files: list, book_dir: str, include_intro: bool = True):
    """
    Create a Jupyter Book _toc.yml using the list of chapter filenames.

    chapter_files: ["chapter1.md", "chapter2.md"]
    book_dir: path to Jupyter Book root ("my-jupyter-book")

    This function will write:
    - format: jb-book
    - root: intro
    - chapters:
        - file: content/chapter1
        - file: content/chapter2
    """

    toc_path = os.path.join(book_dir, "_toc.yml")
    ensure_dir(book_dir)

    # Start TOC structure
    toc_lines = []
    toc_lines.append("format: jb-book")

    if include_intro:
        toc_lines.append("root: intro")
    else:
        toc_lines.append(f"root: content/{chapter_files[0].replace('.md','')}")

    toc_lines.append("")
    toc_lines.append("chapters:")

    # Add each chapter
    for chapter in chapter_files:
        chapter_no_ext = chapter.replace(".md", "")
        toc_lines.append(f"  - file: content/{chapter_no_ext}")

    # Write TOC file
    with open(toc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(toc_lines))


def auto_save_chapter(text: str, book_root: str = "my-jupyter-book") -> str:
    """
    Save a chapter automatically with:
    - timestamp-based filename
    - auto path handling
    - returns filename only
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chapter_{timestamp}.md"

    out_dir = os.path.join(book_root, "content")
    ensure_dir(out_dir)

    save_markdown(text, out_dir, filename)

    return filename
``
