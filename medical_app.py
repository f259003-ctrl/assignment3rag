# medical_app.py
import streamlit as st
import pandas as pd
from medical_rag_pipeline import MedicalRAGSystem
import time

# Page configuration
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("Loading Medical RAG System..."):
        st.session_state.rag_system = MedicalRAGSystem()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🏥 Medical RAG Assistant")
st.markdown("""
This assistant provides evidence-based medical information using the Medical Transcriptions dataset.
**Disclaimer:** This is for informational purposes only. Always consult healthcare professionals for medical advice.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This RAG system uses:
    - Medical Transcriptions Dataset
    - Google Gemini Pro
    - FAISS Vector Store
    - LangChain Framework
    """)
    
    st.header("Example Questions")
    example_questions = [
        "What are common symptoms of diabetes?",
        "How is hypertension treated?",
        "Describe the procedure for a physical examination",
        "What medications are used for asthma?",
        "Explain the recovery process after surgery"
    ]
    
    for q in example_questions:
        if st.button(q):
            st.session_state.user_question = q

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Medical Q&A")
    
    # User input
    user_question = st.text_input(
        "Ask a medical question:",
        value=st.session_state.get('user_question', ''),
        placeholder="e.g., What are the symptoms of pneumonia?"
    )
    
    if st.button("Get Answer") and user_question:
        with st.spinner("Searching medical knowledge base..."):
            start_time = time.time()
            result = st.session_state.rag_system.query(user_question)
            response_time = time.time() - start_time
            
            # Store in chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": result["answer"],
                "sources": result["source_documents"],
                "response_time": response_time
            })
    
    # Display chat history
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {chat['question']}", expanded=i==0):
            st.write("**Answer:**")
            st.write(chat["answer"])
            
            st.write("**Sources:**")
            for j, doc in enumerate(chat["sources"]):
                with st.expander(f"Source {j+1}: {doc.metadata.get('medical_specialty', 'Unknown')}"):
                    st.write(f"Specialty: {doc.metadata.get('medical_specialty', 'N/A')}")
                    st.write(f"Description: {doc.metadata.get('description', 'N/A')}")
                    st.write("Content preview:")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            st.caption(f"Response time: {chat['response_time']:.2f}s")

with col2:
    st.subheader("System Information")
    
    if st.session_state.chat_history:
        latest_chat = st.session_state.chat_history[-1]
        st.metric("Latest Response Time", f"{latest_chat['response_time']:.2f}s")
        st.metric("Sources Retrieved", len(latest_chat["sources"]))
    
    # Statistics
    st.subheader("Statistics")
    st.write(f"Total Queries: {len(st.session_state.chat_history)}")
    
    if st.session_state.chat_history:
        avg_response_time = sum(chat['response_time'] for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.write(f"Average Response Time: {avg_response_time:.2f}s")
    
    # Clear history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.caption("Medical RAG System | Built with LangChain & Gemini API")
