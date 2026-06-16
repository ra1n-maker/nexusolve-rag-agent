from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

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
    
    print("Loading FastEmbed model (No PyTorch required!)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Indexing into Qdrant...")
    qdrant = QdrantVectorStore.from_documents(
        splits,
        embeddings,
        path="./qdrant_db", 
        collection_name="nexusolve_projects"
    )
    
    print(f"Success! Indexed {len(splits)} chunks into the Qdrant database.")

if __name__ == "__main__":
    build_index()