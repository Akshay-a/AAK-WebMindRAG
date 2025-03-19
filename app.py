import streamlit as st
from backend_service import RAGService
from ui_components import UIComponents

def main():
    st.set_page_config(
        page_title="WebMindRAG",
        page_icon="",
        layout="wide"
    )

    # Initialize service and UI components
    rag_service = RAGService()
    ui = UIComponents()

    # Display header and sidebar
    ui.header()
    ui.sidebar()

    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Add Documents", "Search Knowledge Base"])

    with tab1:
        ui.add_document_section(rag_service.process_page_and_children)
        ui.stats_section(rag_service.get_index_stats)

    with tab2:
        ui.search_section(rag_service.search_document, rag_service.query_with_rag)

if __name__ == "main":
    main()

