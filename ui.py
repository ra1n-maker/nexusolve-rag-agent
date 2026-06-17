import streamlit as st
import requests

# Configure the page
st.set_page_config(page_title="Nexusolve Project Intelligence", page_icon="🏭")

st.title("🏭 Nexusolve Agentic RAG")
st.markdown("Ask questions about project timelines, supplier risks, and dependencies.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("E.g., What is causing delays for Project Atlas?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is searching the knowledge base and reasoning..."):
            try:
                # Send request to our FastAPI backend
                # 'api' is the hostname we will define in docker-compose
                response = requests.post("http://api:8000/ask", json={"question": prompt})
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend API. Is it running?")
                response = requests.post("http://api:8000/ask", json={"question": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    # Check if the backend threw an error
                    if "error" in data:
                        st.error(f"Backend Error: {data['error']}")
                    else:
                        answer = data.get("answer", "No answer found.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API HTTP Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend API. Is it running?")
