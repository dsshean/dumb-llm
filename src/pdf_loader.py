import os
from typing import List
import PyPDF2


def load_pdf_text(pdf_path: str) -> str:
    """Extract text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        print(f"Loaded PDF: {os.path.basename(pdf_path)} ({num_pages} pages)")
        return text
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")
        return ""


def load_pdfs_from_directory(directory: str = ".") -> List[str]:
    """Load all PDF files from a directory."""
    texts = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return texts
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file}")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        text = load_pdf_text(pdf_path)
        if text:
            texts.append(text)
    
    return texts


def load_sample_pdfs() -> List[str]:
    """Load the two PDF files in the current directory."""
    pdf_files = ["2509.04664v1.pdf", "0907.5356v1.pdf"]
    texts = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            text = load_pdf_text(pdf_file)
            if text:
                texts.append(text)
        else:
            print(f"PDF not found: {pdf_file}")
    
    return texts


if __name__ == "__main__":
    # Test loading PDFs
    texts = load_sample_pdfs()
    print(f"\nLoaded {len(texts)} PDFs successfully")
    for i, text in enumerate(texts):
        print(f"\nPDF {i+1} preview (first 200 chars):")
        print(text[:200] + "...")