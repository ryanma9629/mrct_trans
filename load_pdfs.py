import os
import re
from dotenv import load_dotenv
import pypdf

# Load environment variables
load_dotenv()

# Define paths
PDF_DIRECTORY = "pdf"
CONTEXT_DATA_DIRECTORY = "context_data"

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning to handle PDF extraction artifacts.
    - Replaces multiple newlines with a single space.
    - Removes lingering hyphenation at line breaks.
    """
    # Join words broken by hyphenation and a newline
    text = re.sub(r'(\w)-(\s*)\n(\s*)(\w)', r'\1\4', text)
    # Replace multiple newlines/spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdfs():
    """
    Loads PDFs from the specified directory, extracts their text content,
    cleans it, and saves it to .txt files in the context_data directory.
    """
    if not os.path.exists(CONTEXT_DATA_DIRECTORY):
        os.makedirs(CONTEXT_DATA_DIRECTORY)
        print(f"Created directory: {CONTEXT_DATA_DIRECTORY}")

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{PDF_DIRECTORY}' directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        chapter_name = os.path.splitext(pdf_file)[0]
        output_txt_path = os.path.join(CONTEXT_DATA_DIRECTORY, f"{chapter_name}.txt")
        
        print(f"Processing {pdf_path}...")

        try:
            # Load PDF using pypdf
            reader = pypdf.PdfReader(pdf_path)
            
            # Concatenate all page content
            full_text = "".join([page.extract_text() for page in reader.pages])
            
            # Clean the extracted text
            cleaned_text = clean_text(full_text)
            
            # Save to a text file
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"Successfully extracted and saved text to {output_txt_path}")

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print("\nAll PDFs have been processed.")
    print(f"Text files are saved in the '{CONTEXT_DATA_DIRECTORY}' directory.")

if __name__ == "__main__":
    extract_text_from_pdfs()
