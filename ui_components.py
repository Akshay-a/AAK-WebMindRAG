import streamlit as st
from typing import Callable, Optional, Dict, Anyclass UIComponents:
    @staticmethod

    def header() -> None:
        """Display application header"""
        st.title("PATHFINDER the Confluence Knowledge Bot")
        st.markdown("---")

@staticmethod
def sidebar() -> None:
    """Display the sidebar navigation"""
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")

@staticmethod
def add_document_section(add_document_func: Callable) -> None:
    """Create UI section for adding documents"""
    st.header("Add Document")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        page_id = st.text_input("Confluence Page ID", placeholder="Enter page ID")
        
    with col2:
        add_button = st.button("Add to Knowledge Base", key="add_button")
    
    if add_button and page_id:
        with st.spinner("Adding document to knowledge base..."):
            result = add_document_func(page_id)
            
            if result["status"] == "success":
                st.success(f"✅ Successfully added '{result['page_title']}' to knowledge base!")
                st.json(result["index_stats"])
            else:
                st.error(f"❌ Error: {result['message']}")
    
    st.markdown("---")

@staticmethod
def search_section(search_func: Callable, query_func: Callable) -> None:
    """Create UI section for searching documents"""
    st.header("Search Knowledge Base")
    
    query = st.text_input("Ask a question", placeholder="What is a Fraud Alert?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_button = st.button("Search", key="search_button")
    
    with col2:
        answer_button = st.button("Get AI Answer", key="answer_button")
    
    if search_button and query:
        with st.spinner("Searching knowledge base..."):
            results = search_func(query)
            
            st.subheader("Search Results")
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1} (Relevance: {1-result['distance']:.2f})"):
                    st.markdown(result['content'])
                    st.caption(f"Section: {result.get('section_context', 'N/A')}")
    
    if answer_button and query:
        with st.spinner("Generating answer..."):
            answer = query_func(query)
            
            st.subheader("AI Answer")
            st.info(answer)
    
    st.markdown("---")

@staticmethod
def stats_section(stats_func: Callable) -> None:
    """Create UI section for displaying database stats"""
    st.header("Knowledge Base Statistics")
    
    if st.button("Refresh Stats"):
        with st.spinner("Loading statistics..."):
            stats = stats_func()
            st.json(stats)

