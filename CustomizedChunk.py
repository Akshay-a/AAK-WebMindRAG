from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Dict, List, Optional, NamedTuple, Tuple, Any, Iterator, Union
import re
from dataclasses import dataclass, field
from collections import deque
import uuid
from ContextInitialization import ContextStack
from DocumentStructure import Section,DocumentAnalysis,analyze_document_structure

@dataclass
class ProcessedElement:
    """Represents a processed element from the HTML document."""
    element_type: str  # 'heading', 'paragraph', 'list', 'table', 'code', 'image', etc.
    content: str  # Textual content
    html: Optional[str] = None  # Original HTML if needed
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[ContextStack] = None
    section_id: Optional[str] = None
    element_id: str = field(default_factory=lambda: f"element-{uuid.uuid4().hex[:8]}")

    # Track element size for chunking
    token_count: int = 0
    char_count: int = 0
    importance: int = 1  # Higher number = more important for chunking decisions

@dataclass
class DocumentChunk:
    """Represents a chunk of the document with context."""
    chunk_id: str
    content: str
    elements: List[ProcessedElement]
    context: ContextStack
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    char_count: int = 0
    section_ids: List[str] = field(default_factory=list)
@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    max_chunk_size: int = 1000  # Maximum chunk size in tokens
    overlap_size: int = 100  # Overlap size in tokens
    min_chunk_size: int = 100  # Minimum chunk size in tokens
    preserve_tables: bool = True  # Keep tables together when possible
    preserve_lists: bool = True  # Keep lists together when possible
    preserve_code_blocks: bool = True  # Keep code blocks together
    include_html: bool = False  # Include original HTML in processed elements
    estimate_tokens: bool = True  # Estimate token count (approximation)
    
def estimate_token_count(text: str) -> int:
    """Approximate token count based on whitespace-splitting heuristic."""
    return len(text.split())

def traverse_elements(element: Union[Tag, NavigableString]) -> Iterator[Union[Tag, NavigableString]]:
    """
    Generator to traverse through all elements in depth-first order.

    Args:
        element: BeautifulSoup Tag or NavigableString element
        
    Yields:
        Each element in the tree
    """
    if isinstance(element, NavigableString):
        yield element
        return

    # First yield the element itself
    yield element

    # Then recursively process its children
    for child in element.children:
        yield from traverse_elements(child)

def update_context_for_element(
    context: ContextStack, 
    element: Union[Tag, NavigableString], 
    section_map: Dict[str, Section]
) -> ContextStack:
    """
    Update context based on the current element and section map.

Args:
    context: Current context stack
    element: Current HTML element being processed
    section_map: Map of section IDs to Section objects
    
    Returns:
        Updated context stack
    """
    # For string elements, no context update needed
    if isinstance(element, NavigableString):
        return context

    # Clone the context to avoid modifying the original
    new_context = context.clone()

    # Check for heading elements
    if element.name and re.match(r'^h[1-6]\$', element.name):
        level = int(element.name[1])
        heading_text = element.get_text(strip=True)
        
        # Find matching section in section map
        matching_section = None
        for section_id, section in section_map.items():
            if section.level == level and section.title == heading_text:
                matching_section = section
                break
        
        if matching_section:
            # Update current section ID
            new_context.current_section_id = matching_section.id
            
            # Update section path by removing any sections at or below current level
            while (new_context.section_path and 
                section_map.get(new_context.section_path[-1]) and 
                section_map[new_context.section_path[-1]].level >= level):
                new_context.section_path.pop()
                if new_context.heading_path:
                    new_context.heading_path.pop()
            
            # Add current section to path
            new_context.section_path.append(matching_section.id)
            new_context.heading_path.append(heading_text)

    # List context updates
    elif element.name == 'ul':
        new_context.list_stack.append({'type': 'unordered', 'level': new_context.current_list_level + 1})
        new_context.current_list_level += 1
        new_context.current_list_type = 'unordered'
    elif element.name == 'ol':
        new_context.list_stack.append({'type': 'ordered', 'level': new_context.current_list_level + 1})
        new_context.current_list_level += 1
        new_context.current_list_type = 'ordered'
    elif element.name == 'li':
        pass  # List item context already handled by ul/ol
    elif new_context.current_list_level > 0 and element.name not in ['ul', 'ol', 'li']:
        # Check if we're exiting a list context
        parent_list_tags = [p.name for p in element.parents if p.name in ['ul', 'ol', 'li']]
        if not parent_list_tags:
            # We've exited all lists
            new_context.list_stack = []
            new_context.current_list_level = 0
            new_context.current_list_type = None

    # Table context updates
    elif element.name == 'table':
        new_context.in_table = True
        new_context.current_row_index = -1
        new_context.current_cell_index = -1
        new_context.table_headers = []
    elif element.name == 'tr':
        new_context.current_row_index += 1
        new_context.current_cell_index = -1
    elif element.name in ['td', 'th']:
        new_context.current_cell_index += 1
        # If this is a header cell in the first row, add to table_headers
        if element.name == 'th' or (new_context.current_row_index == 0 and not new_context.table_headers):
            header_text = element.get_text(strip=True)
            if len(new_context.table_headers) <= new_context.current_cell_index:
                new_context.table_headers.append(header_text)
    elif new_context.in_table and not any(p.name == 'table' for p in element.parents):
        # We've exited the table
        new_context.in_table = False
        new_context.current_row_index = -1
        new_context.current_cell_index = -1

    # Code block context
    if element.name == 'pre' or element.name == 'code' or element.get('class') and 'code' in ' '.join(element.get('class')):
        new_context.in_code_block = True
    elif new_context.in_code_block and not any(p.name in ['pre', 'code'] for p in element.parents):
        new_context.in_code_block = False

    # Blockquote context
    if element.name == 'blockquote':
        new_context.in_blockquote = True
    elif new_context.in_blockquote and not any(p.name == 'blockquote' for p in element.parents):
        new_context.in_blockquote = False

    return new_context

def process_element_by_type(
    element: Union[Tag, NavigableString], 
    context: ContextStack, 
    config: ProcessingConfig
) -> Optional[ProcessedElement]:
    """
    Process an element based on its type, extracting relevant content.

Args:
    element: HTML element to process
    context: Current context stack
    config: Processing configuration
    
    Returns:
        ProcessedElement or None if element should be skipped
    """
    if isinstance(element, NavigableString):
        # Skip empty strings or whitespace-only strings
        text = str(element).strip()
        if not text or element.parent.name in ['script', 'style']:
            return None
            
        # Return text content with parent's context
        return ProcessedElement(
            element_type="text",
            content=text,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(text) if config.estimate_tokens else 0,
            char_count=len(text)
        )

    # Process different element types
    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        heading_level = int(element.name[1])
        heading_text = element.get_text(strip=True)
        
        if not heading_text:
            return None
            
        return ProcessedElement(
            element_type=f"heading-{heading_level}",
            content=heading_text,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(heading_text) if config.estimate_tokens else 0,
            char_count=len(heading_text),
            importance=7 - heading_level  # h1 is more important than h6
        )

    elif element.name == 'p':
        paragraph_text = element.get_text(strip=True)
        
        if not paragraph_text:
            return None
            
        return ProcessedElement(
            element_type="paragraph",
            content=paragraph_text,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(paragraph_text) if config.estimate_tokens else 0,
            char_count=len(paragraph_text),
            importance=2  # Paragraphs are moderately important
        )

    elif element.name == 'table' and element.find('tr'):
        # Process entire table at once to maintain structure
        table_text = []
        
        # Process headers
        headers = []
        header_row = element.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            if header_cells:
                headers = [cell.get_text(strip=True) for cell in header_cells]
                table_text.append(" | ".join(headers))
                table_text.append("-" * (len(" | ".join(headers))))
        
        # Process rows
        for row in element.find_all('tr')[1:] if headers else element.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if any(cells):  # Skip empty rows
                table_text.append(" | ".join(cells))
        
        table_content = "\n".join(table_text)
        
        if not table_content:
            return None
            
        return ProcessedElement(
            element_type="table",
            content=table_content,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(table_content) if config.estimate_tokens else 0,
            char_count=len(table_content),
            importance=4  # Tables are important structural elements
        )

    elif element.name in ['ul', 'ol'] and element.find('li'):
        # Process entire list at once
        list_type = "unordered" if element.name == 'ul' else "ordered"
        list_items = []
        
        for idx, item in enumerate(element.find_all('li', recursive=False)):
            prefix = f"{idx+1}. " if list_type == "ordered" else "• "
            item_text = item.get_text(strip=True)
            if item_text:
                list_items.append(f"{prefix}{item_text}")
        
        list_content = "\n".join(list_items)
        
        if not list_content:
            return None
            
        return ProcessedElement(
            element_type=list_type + "_list",
            content=list_content,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(list_content) if config.estimate_tokens else 0,
            char_count=len(list_content),
            importance=3  # Lists are somewhat important
        )

    elif element.name == 'pre' or (element.name == 'code' and element.parent.name != 'pre'):
        code_text = element.get_text(strip=True)
        
        if not code_text:
            return None
            
        return ProcessedElement(
            element_type="code",
            content=f"```\n{code_text}\n```",
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(code_text) if config.estimate_tokens else 0,
            char_count=len(code_text),
            importance=4  # Code blocks are important
        )

    elif element.name == 'blockquote':
        quote_text = element.get_text(strip=True)
        
        if not quote_text:
            return None
            
        return ProcessedElement(
            element_type="blockquote",
            content=f"> {quote_text}",
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            token_count=estimate_token_count(quote_text) if config.estimate_tokens else 0,
            char_count=len(quote_text),
            importance=3
        )

    elif element.name == 'img':
        alt_text = element.get('alt', '')
        src = element.get('src', '')
        
        content = f"[Image: {alt_text}]" if alt_text else "[Image]"
        
        return ProcessedElement(
            element_type="image",
            content=content,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            metadata={"src": src, "alt": alt_text},
            token_count=3,  # Approximate token count for image placeholder
            char_count=len(content),
            importance=2
        )

    elif element.name == 'a':
        link_text = element.get_text(strip=True)
        href = element.get('href', '')
        
        if not link_text:
            return None
        
        content = link_text
        
        return ProcessedElement(
            element_type="link",
            content=content,
            html=str(element) if config.include_html else None,
            context=context.clone(),
            section_id=context.current_section_id,
            metadata={"href": href},
            token_count=estimate_token_count(content) if config.estimate_tokens else 0,
            char_count=len(content),
            importance=1
        )

    # For container elements that we process individually, return None
    elif element.name in ['div', 'span', 'section', 'article', 'main']:
        # Only process these if they have direct text content not wrapped in other tags
        direct_text = ''.join(str(c) for c in element.contents if isinstance(c, NavigableString)).strip()
        
        if direct_text:
            return ProcessedElement(
                element_type="text",
                content=direct_text,
                html=str(element) if config.include_html else None,
                context=context.clone(),
                section_id=context.current_section_id,
                token_count=estimate_token_count(direct_text) if config.estimate_tokens else 0,
                char_count=len(direct_text),
                importance=1
            )

    # Skip script, style, and hidden elements
    elif element.name in ['script', 'style', 'meta', 'link', 'noscript']:
        return None

    return None  # Skip other elements by default

def create_natural_chunks(
    structured_content: List[ProcessedElement], 
    section_map: Dict[str, Section], 
    config: ProcessingConfig
) -> List[DocumentChunk]:
    """
    Create semantically meaningful chunks from structured content.

Args:
    structured_content: List of processed elements
    section_map: Map of section IDs to Section objects
    config: Processing configuration
    
    Returns:
        List of document chunks
    """
    chunks = []
    current_chunk = {
        "elements": [],
        "token_count": 0,
        "char_count": 0,
        "section_ids": set(),
        "contexts": []
    }

    for element in structured_content:
        # Skip empty elements
        if not element.content.strip():
            continue
        
        # If adding this element would exceed max chunk size and we have content already
        if (current_chunk["token_count"] + element.token_count > config.max_chunk_size and 
            current_chunk["token_count"] >= config.min_chunk_size):
            
            # Create a new chunk from accumulated elements
            if current_chunk["elements"]:
                # Use the most common context as the chunk's context
                chunk_context = current_chunk["contexts"][0] if current_chunk["contexts"] else None
                
                chunk_id = f"chunk-{len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content="\n\n".join(e.content for e in current_chunk["elements"]),
                    elements=current_chunk["elements"],
                    context=chunk_context,
                    token_count=current_chunk["token_count"],
                    char_count=current_chunk["char_count"],
                    section_ids=list(current_chunk["section_ids"]),
                    metadata={
                        "section_titles": [
                            section_map[section_id].title for section_id in current_chunk["section_ids"]
                            if section_id in section_map
                        ]
                    }
                ))
                
                # Reset for new chunk
                current_chunk = {
                    "elements": [],
                    "token_count": 0,
                    "char_count": 0,
                    "section_ids": set(),
                    "contexts": []
                }
        
        # Special handling for important elements
        if element.element_type.startswith("heading-"):
            # Start a new chunk for headings if we have content already
            if current_chunk["elements"] and current_chunk["token_count"] >= config.min_chunk_size:
                # Create chunk from accumulated elements
                chunk_context = current_chunk["contexts"][0] if current_chunk["contexts"] else None
                
                chunk_id = f"chunk-{len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content="\n\n".join(e.content for e in current_chunk["elements"]),
                    elements=current_chunk["elements"],
                    context=chunk_context,
                    token_count=current_chunk["token_count"],
                    char_count=current_chunk["char_count"],
                    section_ids=list(current_chunk["section_ids"]),
                    metadata={
                        "section_titles": [
                            section_map[section_id].title for section_id in current_chunk["section_ids"]
                            if section_id in section_map
                        ]
                    }
                ))
                
                # Reset for new chunk
                current_chunk = {
                    "elements": [],
                    "token_count": 0,
                    "char_count": 0,
                    "section_ids": set(),
                    "contexts": []
                }
        
        # Handle tables - don't split tables if possible
        elif element.element_type == "table" and config.preserve_tables:
            if current_chunk["elements"] and element.token_count + current_chunk["token_count"] > config.max_chunk_size:
                # Create chunk from accumulated elements
                chunk_context = current_chunk["contexts"][0] if current_chunk["contexts"] else None
                
                chunk_id = f"chunk-{len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content="\n\n".join(e.content for e in current_chunk["elements"]),
                    elements=current_chunk["elements"],
                    context=chunk_context,
                    token_count=current_chunk["token_count"],
                    char_count=current_chunk["char_count"],
                    section_ids=list(current_chunk["section_ids"]),
                    metadata={
                        "section_titles": [
                            section_map[section_id].title for section_id in current_chunk["section_ids"]
                            if section_id in section_map
                        ]
                    }
                ))
                
                # Reset for new chunk
                current_chunk = {
                    "elements": [],
                    "token_count": 0,
                    "char_count": 0,
                    "section_ids": set(),
                    "contexts": []
                }
            
            # If table is too large for a single chunk, we have to split it somehow
            if element.token_count > config.max_chunk_size:
                # Add table as its own chunk
                chunk_id = f"chunk-{len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=element.content,
                    elements=[element],
                    context=element.context,
                    token_count=element.token_count,
                    char_count=element.char_count,
                    section_ids=[element.section_id] if element.section_id else [],
                    metadata={"element_type": "table"}
                ))
                continue
        
        # Add element to current chunk
        current_chunk["elements"].append(element)
        current_chunk["token_count"] += element.token_count
        current_chunk["char_count"] += element.char_count
        if element.section_id:
            current_chunk["section_ids"].add(element.section_id)
        if element.context:
            current_chunk["contexts"].append(element.context)

    #add the last chunk if it has content
    if current_chunk["elements"]:
        chunk_context = current_chunk["contexts"][0] if current_chunk["contexts"] else None
        
        chunk_id = f"chunk-{len(chunks)}"
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            content="\n\n".join(e.content for e in current_chunk["elements"]),
            elements=current_chunk["elements"],
            context=chunk_context,
            token_count=current_chunk["token_count"],
            char_count=current_chunk["char_count"],
            section_ids=list(current_chunk["section_ids"]),
            metadata={
                "section_titles": [
                    section_map[section_id].title for section_id in current_chunk["section_ids"]
                    if section_id in section_map
                ]
            }
        ))

    return chunks

