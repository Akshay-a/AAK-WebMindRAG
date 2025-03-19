import os
import numpy as np
import faiss
import pickle
import json
import time
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass, field
from CustomizedChunk import DocumentChunk
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessageTry to import sentence-transformerstry:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("sentence-transformers not available. Debug the issue bro!!")class RAGVectorStore:
    def init(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",  # Default to a small, fast model
        vector_dimension: Optional[int] = None,
        index_path: Optional[str] = "faiss_index",
        metadata_path: Optional[str] = "chunk_metadata.pkl"
    ):
        """
        Initialize the RAG Vector Store using FAISS.

    Args:
        embedding_model_name: Name of the sentence-transformers model to use
        vector_dimension: Dimension of the embedding vectors (determined automatically)
        index_path: Path to save/load the FAISS index
        metadata_path: Path to save/load chunk metadata
    """
    # Initialize embedding model
    if HAVE_SENTENCE_TRANSFORMERS:
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Initialized embedding model: {embedding_model_name} with dimension {self.vector_dimension}")
    else:
        # Fallback to simple embedding function
        self.model = None
        self.vector_dimension = vector_dimension or 384
        print(f"Using fallback embedding function with dimension {self.vector_dimension}")
    
    # FAISS index parameters
    self.index_path = index_path
    self.metadata_path = metadata_path
    
    # Initialize or load FAISS index
    if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
        self.load_index()
    else:
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.chunk_metadata = []
    
    # #Change: Initialize auto_save to True by default
    self._auto_save = True

def embed_texts(self, texts: List[str]) -> np.ndarray:
    """
    Embed a list of text strings using the embedding model.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        Numpy array of embeddings
    """
    if HAVE_SENTENCE_TRANSFORMERS and self.model:
        # Use sentence-transformers model
        return self.model.encode(texts, convert_to_numpy=True)
    else:
        # Use fallback embedding function
        return np.array([self.simple_embedding_function(text) for text in texts], dtype=np.float32)

def set_auto_save(self, enabled):
    """
    Enable or disable auto-saving of the index after adding documents.
    Returns the previous state.
    """
    previous = getattr(self, '_auto_save', True)
    self._auto_save = enabled
    return previous

def simple_embedding_function(self, text: str) -> List[float]:
    """
    This function is not required now..
    A simple embedding function that creates a deterministic embedding from text.
    works as a fallback when API access is unavailable.
    """
    import hashlib
    import random
    
    # Normalize text
    normalized_text = text.lower().strip()
    
    # Create a hash of the text
    hash_obj = hashlib.sha256(normalized_text.encode('utf-8'))
    hash_digest = hash_obj.digest()
    
    # Use the hash to seed a pseudo-random number generator
    random.seed(int.from_bytes(hash_digest[:8], byteorder='big'))
    
    # Generate deterministic vector
    base_vector = [random.uniform(-1, 1) for _ in range(min(100, self.vector_dimension))]
    
    # Extend to desired dimension if needed
    if self.vector_dimension > 100:
        # Repeat and vary the base vector to fill the desired dimension
        result = []
        while len(result) < self.vector_dimension:
            offset = len(result) // 100
            for val in base_vector:
                if len(result) < self.vector_dimension:
                    result.append(val * (1.0 / (1.0 + 0.1 * offset)))
        return result
    else:
        return base_vector[:self.vector_dimension]

# #Change: Updated add_document_chunks to properly handle auto_save and return a dictionary
def add_document_chunks(self, chunks: List[DocumentChunk]) -> Dict:
    """
    Add document chunks to the vector database.
    
    Args:
        chunks: List of DocumentChunk objects 
        
    Returns:
        Dictionary with status info
    """
    if not chunks:
        print("No chunks provided to add to the index.")
        return {"status": "error", "message": "No chunks provided"}
    
    # Extract text content from chunks
    texts = [chunk.content for chunk in chunks]
    
    # Get embeddings for all texts
    try:
        embeddings = self.embed_texts(texts)
        
        # Add vectors to the index
        self.index.add(embeddings)
        
        # Store chunk metadata
        for i, chunk in enumerate(chunks):
            # Convert DocumentChunk to a serializable dictionary
            # We'll store only essential information to avoid serialization issues
            serializable_metadata = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "section_ids": chunk.section_ids,
                # Store section context for better retrieval context
                "section_path": chunk.context.section_path if hasattr(chunk.context, "section_path") else [],
                "heading_path": chunk.context.heading_path if hasattr(chunk.context, "heading_path") else []
            }
            
            self.chunk_metadata.append(serializable_metadata)
        
        # #Change: Handle auto-save logic with more verbose output
        auto_save = getattr(self, '_auto_save', True)
        if auto_save:
            self.save_index()
            print(f"Auto-saved index with {self.index.ntotal} vectors to {self.index_path}")
        
        print(f"Added {len(chunks)} document chunks to the vector database.")
        return {
            "status": "success",
            "chunks_added": len(chunks)
        }
        
    except Exception as e:
        print(f"Error adding chunks to the index: {e}")
        return {"status": "error", "message": str(e)}

def save_index(self) -> None:
    """Save the FAISS index and chunk metadata to disk."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(self.index, self.index_path)
    
    # Save chunk metadata
    with open(self.metadata_path, 'wb') as f:
        pickle.dump(self.chunk_metadata, f)
    
    print(f"Saved index with {self.index.ntotal} vectors to {self.index_path}")

def load_index(self) -> None:
    """Load the FAISS index and chunk metadata from disk."""
    if os.path.exists(self.index_path):
        self.index = faiss.read_index(self.index_path)
        print(f"Loaded index with {self.index.ntotal} vectors from {self.index_path}")
    else:
        print(f"Index file {self.index_path} not found. Creating new index.")
        self.index = faiss.IndexFlatL2(self.vector_dimension)
    
    if os.path.exists(self.metadata_path):
        with open(self.metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
    else:
        print(f"Metadata file {self.metadata_path} not found. Creating empty metadata.")
        self.chunk_metadata = []

def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks based on a query.
    
    Args:
        query: Search query string
        k: Number of results to return
    
    Returns:
        List of dictionaries containing content and metadata
    """
    # Embed the query
    query_embedding = np.array([self.embed_texts([query])[0]], dtype=np.float32)
    
    # Search the index
    if self.index.ntotal == 0:
        print("Warning: Index is empty, no results to return.")
        return []
        
    distances, indices = self.index.search(query_embedding, k=min(k, self.index.ntotal))
    
    # Retrieve corresponding chunks
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(self.chunk_metadata) and idx >= 0:
            # Add distance score to the result
            result = dict(self.chunk_metadata[idx])
            result["distance"] = float(distances[0][i])
            
            # Format section context for better readability
            if "heading_path" in result:
                result["section_context"] = " > ".join(result["heading_path"])
            
            results.append(result)
    
    return results

def query_with_rag(self, query: str, k: int = 3) -> str:
    """
    Perform RAG-based Q&A using the vector store.
    
    Args:
        query: User query string
        k: Number of context chunks to retrieve
        
    Returns:
        Response string (either from LLM or a fallback message)
    """
    # Start timing
    start_time = time.time()
    
    # Search for relevant context
    search_results = self.search(query, k=k)
    
    if not search_results:
        return "No relevant information found in the knowledge base."
    
    # Extract context from search results with section information
    contexts = []
    for result in search_results:
        # Add section context if available
        if "section_context" in result and result["section_context"]:
            section_info = f"Section: {result['section_context']}\n"
        else:
            section_info = ""
            
        # Format the context with section info
        contexts.append(f"{section_info}{result['content']}")
    
    # Join all contexts with clear separators
    context_str = "\n\n---\n\n".join(contexts)
    
    # Since we may not have access to Bedrock LLM, provide the context
    execution_time = time.time() - start_time
    
    # If no LLM access, return the retrieved context
    return_msg = f"Retrieved {len(search_results)} relevant passages in {execution_time:.2f} seconds.\n\n"
    return_msg += "Here are the most relevant passages:\n\n"
    return_msg += context_str
    
    return return_msg

# #Change: Completely revised to properly handle batches and ensure the index is saved
def batch_add_document_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> Dict:
    """
    Add a large number of document chunks in batches to avoid memory issues.
    
    Args:
        chunks: List of DocumentChunk objects
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Dictionary with status info
    """
    # Disable auto-saving during batch processing for better performance
    original_auto_save = self.set_auto_save(False)
    
    total_chunks_added = 0
    batch_count = (len(chunks) - 1) // batch_size + 1
    
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            result = self.add_document_chunks(batch)
            
            if result["status"] == "success":
                total_chunks_added += result.get("chunks_added", 0)
            
            print(f"Added batch {i//batch_size + 1}/{batch_count}")
        
        # #Change: Always save after all batches are processed, regardless of auto_save setting
        self.save_index()
        print(f"Batch processing complete. Saved index with {self.index.ntotal} total vectors.")
        
        # Restore original auto_save setting
        self.set_auto_save(original_auto_save)
        
        return {
            "status": "success", 
            "chunks_added": total_chunks_added
        }
    
    except Exception as e:
        # #Change: Make sure to save what we have so far in case of an error
        self.save_index()
        # Restore original auto_save setting
        self.set_auto_save(original_auto_save)
        
        print(f"Error in batch processing: {e}")
        return {
            "status": "error",
            "message": str(e),
            "chunks_added": total_chunks_added
        }

# #Change: Completely revised to ensure the latest data is loaded from disk
def get_index_stats(self) -> Dict[str, Any]:
    """
    Get statistics about the current index.
    Ensures the latest data is loaded from disk, even if another process modified it.
    
    Returns:
        Dictionary containing index statistics
    """
    # Always reload the index and metadata directly from disk to get the latest state
    # This bypasses any cached versions that might exist in memory
    if os.path.exists(self.index_path):
        self.index = faiss.read_index(self.index_path)
        print(f"Refreshed stats: Loaded index with {self.index.ntotal} vectors from {self.index_path}")
    else:
        print(f"Warning: Index file {self.index_path} not found when refreshing stats.")
        
    if os.path.exists(self.metadata_path):
        with open(self.metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        print(f"Refreshed stats: Loaded metadata with {len(self.chunk_metadata)} chunks")
    else:
        print(f"Warning: Metadata file {self.metadata_path} not found when refreshing stats.")
        
    # Count sections represented in the index
    unique_sections = set()
    for chunk in self.chunk_metadata:
        if "section_ids" in chunk:
            unique_sections.update(chunk["section_ids"])
    
    stats = {
        "vector_count": self.index.ntotal,
        "vector_dimension": self.vector_dimension,
        "chunk_count": len(self.chunk_metadata),
        "unique_sections": len(unique_sections),
        "index_type": type(self.index).__name__,
        "embedding_model": "sentence-transformers" if HAVE_SENTENCE_TRANSFORMERS else "fallback",
        "last_refreshed": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Include file paths to help with debugging
        "index_path": self.index_path,
        "metadata_path": self.metadata_path
    }
    
    return stats

def process_with_llm(
    self,
    query: str,
    retrieved_context: str,
    task_type: str = "answer",  # Options: "answer", "summarize", "analyze"
    model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0") -> str:
    """
    Process retrieved context with an LLM to generate an answer, summary, or analysis.
    
    Args:
        query: The original user query
        retrieved_context: The context retrieved from vector search
        task_type: The type of processing to perform ("answer", "summarize", "analyze")
        model_id: The model ID to use (defaults to Claude 3.5 Sonnet)
    
    Returns:
        The LLM's response
    """
    # First try to use AWS Bedrock if available
    try:
        # Initialize Bedrock client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # Change to your region if needed
        )
        
        # Initialize LLM
        llm = ChatBedrock(
            model_id=model_id,
            client=bedrock_runtime,
            streaming=False
        )
        
        # Create appropriate system prompts based on task type
        if task_type.lower() == "summarize":
            system_prompt = (
                "You are an AI assistant specialized in summarizing information. "
                "Summarize the provided context clearly and concisely, highlighting the most important points. "
                "Focus on accuracy and clarity."
            )
            human_prompt = f"Please summarize the following information:\n\n{retrieved_context}"
            
        elif task_type.lower() == "analyze":
            system_prompt = (
                "You are an AI assistant specialized in deep analysis. "
                "Analyze the provided context thoroughly, identifying key insights, patterns, implications, and connections. "
                "Provide a detailed analysis with supporting evidence from the text."
            )
            human_prompt = f"Please analyze the following information in depth:\n\n{retrieved_context}"
            
        else:  # Default to "answer"
            system_prompt = (
                "You are a helpful AI assistant. Answer the question accurately based on the provided context. "
                "If the context doesn't contain relevant information to answer the question, acknowledge this limitation. "
                "Be concise, direct, and factual in your response."
            )
            human_prompt = f"Question: {query}\n\nContext:\n{retrieved_context}\n\nAnswer:"
        
        # Create message list
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Get response from Bedrock
        response = llm.invoke(
            input=messages
        )

        return response.content
        
    except Exception as e:
        print(f"Error using AWS Bedrock: {e}")
        return f"Unable to access LLM services for analysis. Here's the retrieved information:\n\n{retrieved_context}"

