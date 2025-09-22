import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Define paths
PDF_DIRECTORY = "pdf"
CHROMA_DB_DIRECTORY = "chroma_db"

def load_pdfs_to_chromadb():
    """
    Loads PDFs from the specified directory, splits them into chunks,
    and stores them in a ChromaDB vector database.
    """
    # Initialize Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )

    # Initialize ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIRECTORY,
        embedding_function=embeddings
    )

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        print(f"Processing {pdf_path}...")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Extract chapter name from filename (e.g., "Chapter11.pdf" -> "Chapter11")
        chapter_name = os.path.splitext(pdf_file)[0]
        
        # Update metadata to ensure consistent source field
        for doc in documents:
            doc.metadata['source'] = chapter_name
            doc.metadata['filename'] = pdf_file
            doc.metadata['filepath'] = pdf_path
        
        print(f"Updated metadata for {len(documents)} documents with source='{chapter_name}'")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Ensure chunks inherit the metadata
        for chunk in chunks:
            chunk.metadata['source'] = chapter_name
            chunk.metadata['filename'] = pdf_file
            chunk.metadata['filepath'] = pdf_path

        # Add chunks to ChromaDB
        vectorstore.add_documents(chunks)
        print(f"Finished processing {pdf_file} - added {len(chunks)} chunks with source='{chapter_name}'.")

    print("All PDFs have been processed and stored in ChromaDB.")

if __name__ == "__main__":
    load_pdfs_to_chromadb()
