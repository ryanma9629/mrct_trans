import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def debug_chroma_metadata():
    """Debug the metadata structure in ChromaDB."""
    
    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
        
        # Initialize ChromaDB
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        
        # Get all documents
        all_docs = vectorstore.get()
        print(f"Total documents in ChromaDB: {len(all_docs['ids'])}")
        
        # Check first few documents' metadata
        print("\nFirst 10 documents metadata:")
        for i in range(min(10, len(all_docs['ids']))):
            doc_id = all_docs['ids'][i]
            metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            print(f"Document {i+1}: ID={doc_id}")
            print(f"  Metadata: {metadata}")
            if all_docs['documents']:
                content_preview = all_docs['documents'][i][:100] + "..." if len(all_docs['documents'][i]) > 100 else all_docs['documents'][i]
                print(f"  Content preview: {content_preview}")
            print()
        
        # Check specifically for Chapter 11 content
        print("\nSearching for documents containing 'multiregional clinical trial'...")
        test_query = "multiregional clinical trial MRCT"
        
        # Try similarity search without filter first
        similar_docs = vectorstore.similarity_search(test_query, k=10)
        print(f"Found {len(similar_docs)} similar documents without filter:")
        
        for i, doc in enumerate(similar_docs):
            print(f"Document {i+1}:")
            print(f"  Metadata: {doc.metadata}")
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"  Content: {content_preview}")
            print()
        
        # Try different filter variations
        print("\nTesting different filter variations...")
        
        filter_variations = [
            {"source": "Chapter11"},
            {"source": "Chapter11.pdf"},
            {"source": "pdf/Chapter11.pdf"},
            {"source": "pdf\\Chapter11.pdf"},
        ]
        
        for filter_var in filter_variations:
            try:
                filtered_docs = vectorstore.similarity_search(test_query, k=5, filter=filter_var)
                print(f"Filter {filter_var}: Found {len(filtered_docs)} documents")
                if filtered_docs:
                    for doc in filtered_docs[:2]:  # Show first 2
                        print(f"  Metadata: {doc.metadata}")
                        print(f"  Content preview: {doc.page_content[:100]}...")
            except Exception as e:
                print(f"Filter {filter_var}: ERROR - {e}")
        
    except Exception as e:
        print(f"Error debugging ChromaDB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chroma_metadata()