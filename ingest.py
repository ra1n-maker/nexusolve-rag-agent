from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse

def build_index():
    print("Loading documents...")
    text_loader = TextLoader("data/meeting_notes_apr12.txt")
    csv_loader = CSVLoader("data/rfq_tracker.csv")
    
    docs = text_loader.load() + csv_loader.load()
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    
    print("Loading Dense and Sparse Models for Hybrid Search...")
    # Dense model for semantic meaning
    dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # Sparse model for exact keyword matching (BM25)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    print("Indexing into Qdrant using Hybrid Mode...")
    qdrant = QdrantVectorStore.from_documents(
        splits,
        dense_embeddings,
        sparse_embedding=sparse_embeddings,
        path="./qdrant_db", 
        collection_name="nexusolve_projects",
        retrieval_mode="hybrid" # Enables Reciprocal Rank Fusion (RRF)
    )
    
    print(f"Success! Indexed {len(splits)} chunks into the Qdrant hybrid database.")

if __name__ == "__main__":
    build_index()