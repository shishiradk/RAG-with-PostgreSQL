import streamlit as st
import requests
import pandas as pd
import os

# Configuration - handle both local and production
API_BASE_URL = os.getenv("API_URL", "http://localhost:8001")

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, None

def ask_question(question, limit=3):
    try:
        payload = {"question": question, "limit": limit}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_stats():
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    st.set_page_config(
        page_title="RAG System Demo - PostgreSQL + OpenAI",
        layout="wide"
    )
    
    st.title("RAG System Demo")
    st.markdown("### Retrieval-Augmented Generation with PostgreSQL & OpenAI")
    
    # Check API connection first
    with st.spinner("Checking system status..."):
        api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("Backend Server Not Connected")
        st.info(f"Trying to connect to: {API_BASE_URL}")
        st.write("If this is a production deployment, make sure:")
        st.write("- Backend server is running")
        st.write("- Environment variables are set")
        st.write("- Ports 8000 and 8501 are accessible")
        return
    
    # Show connection status
    if health_data and health_data.get('database_connected'):
        st.success("Backend Server & Database Connected")
    else:
        st.warning("Backend Server Running (Database Connection Issues)")
        st.info("The API server is running but cannot connect to the database.")

    # Project Overview
    st.markdown("---")
    st.header("Live Demo - Ask Questions")
    
    # Pre-defined example questions
    example_questions = [
        "What are your shipping options?",
        "How can I track my order?",
        "What is your return policy?",
        "Do you offer international shipping?",
        "What payment methods do you accept?"
    ]
    
    # Quick question buttons
    st.markdown("Try these example questions:")
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(question, key=f"btn_{i}"):
                st.session_state.demo_question = question
    
    # Question input
    question = st.text_input(
        "Or ask your own question:",
        value=st.session_state.get('demo_question', ''),
        placeholder="Type your question here..."
    )
    
    if st.button("Get Answer", type="primary") and question:
        with st.spinner("Searching knowledge base and generating answer..."):
            response = ask_question(question)
            
            if response:
                # Display answer
                st.success("Answer Generated Successfully!")
                
                st.markdown("### Answer")
                st.info(response["answer"])
                
                # Context information
                col1, col2 = st.columns(2)
                with col1:
                    context_status = "Sufficient Context" if response["enough_context"] else "Limited Context"
                    st.metric("Context Quality", context_status)
                
                with col2:
                    st.metric("Sources Used", len(response["sources"]))
                
                # Sources
                if response["sources"]:
                    with st.expander(f"View Source Documents ({len(response['sources'])})"):
                        for i, source in enumerate(response["sources"]):
                            st.markdown(f"**Source {i+1}**")
                            st.write(source["content"])
                            if source.get('category'):
                                st.caption(f"Category: {source['category']}")
                            st.markdown("---")
                
                # Thought process
                with st.expander("AI Thought Process"):
                    for thought in response["thought_process"]:
                        st.write(f"- {thought}")
                        
            else:
                st.error("Failed to generate answer. The system might be processing or the backend might be unavailable.")
    
    # System Information
    st.markdown("---")
    st.header("System Information")
    
    stats = get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats["total_documents"])
        
        with col2:
            st.metric("Vector Database", stats["vector_store"])
            
        with col3:
            st.metric("Embedding Model", "text-embedding-3-small")
            
        with col4:
            st.metric("Dimensions", stats["embedding_dimensions"])
        
        # Category distribution
        if stats["category_distribution"]:
            st.subheader("Knowledge Base Categories")
            categories = list(stats["category_distribution"].keys())
            counts = list(stats["category_distribution"].values())
            
            chart_data = pd.DataFrame({
                "Category": categories,
                "Documents": counts
            })
            
            st.bar_chart(chart_data.set_index("Category"))
    else:
        st.info("System statistics not available")

    # Technical Architecture
    st.markdown("---")
    st.header("Technical Architecture")
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("**Backend Stack**")
        st.markdown("""
        - FastAPI (Python)
        - PostgreSQL + pgvector
        - OpenAI GPT-4 & Embeddings
        - Streamlit UI
        """)
    
    with arch_col2:
        st.markdown("**RAG Pipeline**")
        st.markdown("""
        1. Question → Vector Embedding
        2. Semantic Search → Similar Documents
        3. Context Retrieval → Relevant Information  
        4. Answer Synthesis → AI Response
        5. Source Attribution → Reference Documents
        """)

if __name__ == "__main__":
    main()