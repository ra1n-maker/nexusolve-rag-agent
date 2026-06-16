import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# TODO: Paste your free Groq API key here
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

def run_agent():
    print("Loading embedding model and connecting to Qdrant...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="nexusolve_projects",
        path="./qdrant_db",
    )
    
    # 1. Create the Retrieval Tool (Increased k=4 to get more context)
    retriever = qdrant.as_retriever(search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="project_knowledge_base",
        description="Search for project risks, supplier delays, meeting notes, and dependencies."
    )
    tools = [retriever_tool]

    # 2. Initialize the LLM (Using Llama-3 70B via Groq)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # 3. Create the LangGraph Agent (Remove the state_modifier argument)
    agent_executor = create_react_agent(llm, tools)

    # 4. Ask the complex multi-hop question
    system_prompt = """You are a Manufacturing Project Intelligence Agent. 
    Use the provided tools to search the knowledge base. 
    Always cite your sources by referencing the document name or row number from the context."""
    
    query = "What procurement risks are affecting Project Atlas, specifically regarding Supplier X? Combine information from the CSV and the meeting notes."
    
    print(f"\nAsking Agent: '{query}'\n")
    print("Agent is thinking (using tools)...\n")

    # 5. Run the graph, passing the system prompt directly into the messages list
    response = agent_executor.invoke({
        "messages": [
            ("system", system_prompt),
            ("user", query)
        ]
    })
    
    print("--- Final Answer ---")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    run_agent()