import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# TODO: Paste your Groq API key here again!
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# 1. Initialize FastAPI
app = FastAPI(title="Nexusolve RAG API", version="1.0")

# 2. Define the expected request body
class QueryRequest(BaseModel):
    question: str

# 3. Setup the Agent (Run once when server starts)
print("Initializing Agent Engine...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="nexusolve_projects",
    path="./qdrant_db",
)

retriever = qdrant.as_retriever(search_kwargs={"k": 4})
retriever_tool = create_retriever_tool(
    retriever,
    name="project_knowledge_base",
    description="Search for project risks, supplier delays, meeting notes, and dependencies."
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
agent_executor = create_react_agent(llm, [retriever_tool])

system_prompt = """You are a Manufacturing Project Intelligence Agent. 
Use the provided tools to search the knowledge base. 
Always cite your sources by referencing the document name or row number from the context."""

# 4. Create the API Endpoint
@app.post("/ask")
def ask_agent(request: QueryRequest):
    try:
        response = agent_executor.invoke({
            "messages": [
                ("system", system_prompt),
                ("user", request.question)
            ]
        })
        return {"answer": response["messages"][-1].content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "Agent API is running!"}