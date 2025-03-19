import os
import json
import warnings
from typing import List, Optional
from atlassian import Confluence
from bs4 import BeautifulSoup
#Import your existing modules
from ContextInitialization import ContextStack, initialize_context_stack
from DocumentStructure import analyze_document_structure
from CustomizedChunk import update_context_for_element, ProcessingConfig, DocumentChunk, create_natural_chunks, traverse_elements, process_element_by_type
from FaissDB import RAGVectorStore
class RAGService:
    def init(self,rag_store=None, configure_proxy=True):
        # Set up environment
        if configure_proxy:
            PROXY_SERVER = ""
            NO_PROXY = "169.254.169.254,[fd00:ec2::254]" # No proxy for AWS metadata service
            os.environ['HTTP_PROXY'] = PROXY_SERVER
            os.environ['HTTPS_PROXY'] = PROXY_SERVER
            os.environ['NO_PROXY'] = NO_PROXY
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            warnings.filterwarnings("ignore")

        # Initialize vector store
        self.rag_store = rag_store if rag_store is not None else RAGVectorStore()
        
        # Confluence connection info
        self.confluence_url = os.environ.get("CONFLUENCE_URL", "https://apsingi.atlassian.net/")
        self.api_token = os.environ.get("CONFLUENCE_API_TOKEN","")#your API key to access 
        self.username = os.environ.get("CONFLUENCE_USERNAME","apsingiakshay46@gmail.com")
        
        # Initialize Confluence connection
        self.confluence = Confluence(
            url=self.confluence_url,
            token=self.api_token,
            verify_ssl=False
        )

def process_confluence_html(self, html_content: str, config: Optional[ProcessingConfig] = None) -> List[DocumentChunk]:
    """
    Process HTML content into optimized chunks for RAG applications.
    """
    if config is None:
        config = ProcessingConfig()
    
    # Step 1: Analyze document structure
    document_analysis = analyze_document_structure(html_content)
    document_title = document_analysis.title
    section_map = document_analysis.sections
    main_content = document_analysis.content
    
    # Step 2: Initialize context
    base_context = initialize_context_stack()
    base_context.document_title = document_title
    
    # Step 3 & 4: Process elements and create structured content
    structured_content = []
    current_context = base_context
    
    # Track all elements we've visited to avoid duplicates
    processed_element_ids = set()
    
    for element in traverse_elements(main_content):
        # Update context based on current element
        current_context = update_context_for_element(current_context, element, section_map)
        
        # Process element based on its type
        processed_element = process_element_by_type(element, current_context, config)
        
        # Add to structured content if relevant and not a duplicate
        if processed_element is not None:
            element_content_hash = hash(processed_element.content)
            if element_content_hash not in processed_element_ids:
                processed_element_ids.add(element_content_hash)
                structured_content.append(processed_element)
    
    # Create natural chunks from structured content
    chunks = create_natural_chunks(structured_content, section_map, config)
    
    return chunks

def add_document(self, page_id: str) -> dict:
    """
    Add a Confluence page to the RAG database
    Returns stats about the operation
    """
    # Get page content from Confluence
    #page = self.confluence.get_page_by_id(page_id, expand='body.storage')
    #if not page:
    #    return {"status": "error", "message": f"Page with ID {page_id} not found."}
    
    #page_title = page['title'].replace('/','&').replace(' ','_')
    #html_content = page['body']['storage']['value']
    with open("C:/Project/BedRock/sampleTableHtml.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    # Process the HTML content
    config = ProcessingConfig(
        max_chunk_size=1000,
        overlap_size=100,
        min_chunk_size=200,
        preserve_tables=True,
        preserve_lists=True,
        preserve_code_blocks=True
    )
    
    chunks = self.process_confluence_html(html_content, config)
    
    # Add chunks to vector store
    self.rag_store.add_document_chunks(chunks)
    
    # Return stats
    stats = self.rag_store.get_index_stats()
    return {
        "status": "success",
        "page_title": "empty", 
        "chunks_added": len(chunks),
        "index_stats": stats
    }
# #Change: Updated to use the existing RAG store instance rather than creating a new one
def process_page_and_children(self, page_id: str):

    # Use the existing RAG store instead of creating a new one
    rag_store = self.rag_store
    
    # Disable auto-saving of the index during batch processing
    if hasattr(rag_store, 'set_auto_save'):
        original_auto_save = rag_store.set_auto_save(False)
    
    # Stats to track progress
    stats = {
        "pages_processed": 0,
        "pages_failed": 0,
        "total_chunks": 0,
        "processed_pages": []
    }
    
    # Collect all pages first (parent and children)
    try:
        all_pages = [{"id": page_id, "is_parent": True}]
        child_pages = self.get_all_child_pages(self.confluence, page_id)
        
        # Add children to the processing list
        for child in child_pages:
            all_pages.append({
                "id": child['id'],
                "title": child['title'],
                "is_parent": False
            })
        
        print(f"Found {len(all_pages)-1} child pages under page ID {page_id}")
    except Exception as e:
        print(f"Error retrieving child pages: {str(e)}")
        # Even if we can't get children, try to process the parent
        all_pages = [{"id": page_id, "is_parent": True}]
    
    # Collect all chunks first before adding to the database
    all_chunks = []
    page_title = "Multiple Pages"  # Default title
    
    # Process each page and collect chunks
    for page_info in all_pages:
        page_id = page_info["id"]
        page_type = "parent" if page_info.get("is_parent", False) else "child"
        page_title = page_info.get("title", f"Page {page_id}")
        
        print(f"Processing {page_type} page: {page_title} (ID: {page_id})")
        
        result = self.process_single_page_chunks(
            self.confluence, 
            page_id, 
            self.confluence_url
        )
        
        if result.get("success", False):
            stats["pages_processed"] += 1
            stats["total_chunks"] += len(result["chunks"])
            all_chunks.extend(result["chunks"])
            stats["processed_pages"].append({
                "id": page_id,
                "title": result["title"],
                "chunks": len(result["chunks"])
            })
            
            # Store parent page title
            if page_info.get("is_parent", False):
                page_title = result.get("title", page_title)
        else:
            stats["pages_failed"] += 1
            print(f"Failed to process page {page_id}: {result.get('error', 'Unknown error')}")
    
    # Add all chunks to the vector store in a single batch
    batch_result = {"status": "no_chunks", "chunks_added": 0}
    if all_chunks:
        print(f"\nAdding all {len(all_chunks)} chunks from {stats['pages_processed']} pages to the vector database...")
        # Use batch_add_document_chunks which ensures saving happens at the end
        batch_result = rag_store.batch_add_document_chunks(all_chunks)
        if batch_result.get("status") == "success":
            print(f"Successfully added {batch_result.get('chunks_added', 0)} chunks to the database.")
        else:
            print(f"Error during batch processing: {batch_result.get('message', 'Unknown error')}")
    else:
        print("No chunks were generated from any pages.")
    
    # Re-enable auto-save if it was disabled
    if hasattr(rag_store, 'set_auto_save'):
        rag_store.set_auto_save(original_auto_save)
    
    # Get the latest stats after adding all chunks
    current_stats = rag_store.get_index_stats()
    
    return {
        "status": batch_result.get("status", "no_chunks") if batch_result.get("status") != "error" else "partial_success",
        "page_title": page_title,
        "chunks_added": batch_result.get("chunks_added", 0),
        "index_stats": current_stats
    }

def process_single_page_chunks(self,confluence, page_id, confluence_url):
    """Process a single Confluence page and return its chunks (without adding to RAG store).
    
    Args:
        confluence: Confluence client instance
        page_id: ID of the page to process
        confluence_url: Base URL of the Confluence instance
        process_confluence_html: Function to process HTML content
        
    Returns:
        Dictionary with processing result including chunks
    """
    try:
        # Get page content
        page = confluence.get_page_by_id(page_id, expand='body.storage')
        if not page:
            return {
                "success": False,
                "error": f"Page with ID {page_id} not found."
            }
        
        page_title = page['title'].replace('/','&').replace(' ','_')
        page_url = f"{confluence_url}/plugins/viewsource/viewpagesrc.action?pageId={page_id}"
        html_content = page['body']['storage']['value']
            
        config = ProcessingConfig(
            max_chunk_size=1000,
            overlap_size=100,
            min_chunk_size=200,
            preserve_tables=True,
            preserve_lists=True,
            preserve_code_blocks=True
        )
        
        chunks = self.process_confluence_html(html_content, config)
        
        print(f"Generated {len(chunks)} chunks from {page_title}")
        
        # Add page source metadata to each chunk
        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata['source_page_id'] = page_id
            chunk.metadata['source_page_title'] = page_title
            chunk.metadata['source_url'] = page_url
        
        return {
            "success": True,
            "title": page_title,
            "chunks": chunks
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def process_single_page(self,confluence, page_id, confluence_url,  rag_store):
    """Process a single Confluence page and add its content to the RAG store.
    
    Args:
        confluence: Confluence client instance
        page_id: ID of the page to process
        confluence_url: Base URL of the Confluence instance
        process_confluence_html: Function to process HTML content
        rag_store: RAG vector store instance
        
    Returns:
        Dictionary with processing result
    """
    try:
        # Get page content
        page = confluence.get_page_by_id(page_id, expand='body.storage')
        if not page:
            return {
                "success": False,
                "error": f"Page with ID {page_id} not found."
            }
        
        page_title = page['title'].replace('/','&').replace(' ','_')
        page_url = f"{confluence_url}/plugins/viewsource/viewpagesrc.action?pageId={page_id}"
        html_content = page['body']['storage']['value']
            
        config = ProcessingConfig(
            max_chunk_size=800,
            overlap_size=100,
            min_chunk_size=150,
            preserve_tables=True,
            preserve_lists=True,
            preserve_code_blocks=True
        )
        
        chunks = self.process_confluence_html(html_content, config)
        
        print(f"Generated {len(chunks)} chunks from {page_title}")
        
        # Add page source metadata to each chunk
        for chunk in chunks:
            if 'metadata' not in chunk.__dict__:
                chunk.metadata = {}
            chunk.metadata['source_page_id'] = page_id
            chunk.metadata['source_page_title'] = page_title
            chunk.metadata['source_url'] = page_url
        
        # Add chunks to the vector store
        rag_store.add_document_chunks(chunks)
        
        return {
            "success": True,
            "title": page_title,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def get_all_child_pages(self,confluence, parent_page_id):
    """Get all child pages recursively under a parent page.
    
    Args:
        confluence: Confluence client instance
        parent_page_id: ID of the parent page
        
    Returns:
        List of all child pages (each page is a dict with id, title, etc.)
    """
    all_pages = []
    
    # Use the immediate_children parameter to get direct children first
    children = confluence.get_child_pages(parent_page_id)
    
    # Sometimes the API returns a single dict instead of a list for single child
    if isinstance(children, dict):
        children = [children]
    
    # Add each child to our list and get their children
    for child in children:
        all_pages.append(child)
        
        # Recursively get grandchildren
        grandchildren = self.get_all_child_pages(confluence, child['id'])
        all_pages.extend(grandchildren)
    
    return all_pages

def search_document(self, query: str, k: int = 5) -> List[dict]:
    """Search for content in the vector store"""
    results = self.rag_store.search(query, k=k)
    return results

def query_with_rag(self, query: str) -> str:
    """Perform a RAG query and return the answer"""
    context_from_faiss_db = self.rag_store.query_with_rag(query)
    
    answer = self.rag_store.process_with_llm(
        query=query,
        retrieved_context=context_from_faiss_db,
        task_type="answer"
    )
    return answer
    
def get_index_stats(self) -> dict:
    """Get statistics about the current index"""
    return self.rag_store.get_index_stats()

