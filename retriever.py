from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

def test_retrieval():
    print("Loading FastEmbed model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Connecting to local Qdrant database...")
    # Reconnect to the database folder we just created
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="nexusolve_projects",
        path="./qdrant_db",
    )
    
    # Create a retriever that fetches the top 2 most relevant chunks
    retriever = qdrant.as_retriever(search_kwargs={"k": 2})
    
    # A complex question that requires knowing both the meeting notes and the CSV
    query = "What procurement risks are affecting Project Atlas, specifically regarding Supplier X?"
    print(f"\nSearching for: '{query}'\n")
    
    results = retriever.invoke(query)
    
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata.get('source')}\n")

if __name__ == "__main__":
    test_retrieval()