from bs4 import BeautifulSoup,Tag
from typing import Dict, List, Optional, NamedTuple, Tuple, Any
import re
from dataclasses import dataclass, field

@dataclass
class Section:
    id: str
    level: int
    title: str
    start_pos: int
    end_pos: Optional[int] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

@dataclass
class DocumentAnalysis:
    title: str
    content: Any  # BeautifulSoup object for the main content
    sections: Dict[str, Section]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
def get_main_content_area(soup: BeautifulSoup) -> Tag:
    """
    Finds the main content area in a Confluence HTML document.

    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Tag containing the main content
    """
    # Confluence typically uses specific div IDs/classes for content
    content_selectors = [
        {"id": "main-content"},
        {"class": "wiki-content"},
        {"id": "content"},
        {"class": "confluence-content"}
    ]

    for selector in content_selectors:
        content = soup.find("div", selector)
        if content:
            return content
            
        # Fallback to body if no Confluence-specific container found
    return soup.body or soup

def analyze_document_structure(html_content: str) -> DocumentAnalysis:
    """
    Analyze the structure of a Confluence HTML document.

    Args:
        html_content: The HTML content as a string
        
    Returns:
        DocumentAnalysis object containing document title, sections map, and main content
    """
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract document title
        title_element = soup.find('title') or soup.find('h1')
        document_title = title_element.get_text(strip=True) if title_element else "Untitled Document"
        
        # Find main content area - adapt based on Confluence HTML structure
        main_content = get_main_content_area(soup)
        
        if not main_content:
            raise ValueError("Could not identify main content area in HTML")
        
        # Initialize sections dictionary and section stack
        sections = {}
        section_stack = []
        
        # Generate unique ID for root/document section
        root_id = "root"
        sections[root_id] = Section(
            id=root_id, 
            level=0, 
            title=document_title, 
            start_pos=0, 
            end_pos=None,
            parent_id=None
        )
        section_stack.append(sections[root_id])
        
        # Find all headings in the document
        heading_tags = main_content.find_all(re.compile('^h[1-6]\$'))
        
        for position, heading in enumerate(heading_tags):
            # Extract heading level (h1 -> 1, h2 -> 2, etc.)
            level = int(heading.name[1])
            
            # Extract heading text
            heading_text = heading.get_text(strip=True)
            
            # Create unique ID for this section
            section_id = f"section-{len(sections)}"
            
            # Pop sections from stack if current heading is of same or lower level
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
            
            # Get parent section (if any)
            parent_id = section_stack[-1].id if section_stack else None
            
            # Create new section
            new_section = Section(
                id=section_id,
                level=level,
                title=heading_text,
                start_pos=position,
                parent_id=parent_id
            )
            
            # Add to sections dictionary
            sections[section_id] = new_section
            
            # Update parent's children list
            if parent_id:
                sections[parent_id].children.append(section_id)
            
            # Add to section stack
            section_stack.append(new_section)
            
            # Update end position of previous sections at same level
            if parent_id:
                sections[parent_id].end_pos = position
            
        # Return document analysis object
        return DocumentAnalysis(
            title=document_title,
            content=main_content,
            sections=sections
        )
        
    except Exception as e:
        # Log the error and re-raise with more context
        import logging
        logging.error(f"Error analyzing document structure: {str(e)}")
        raise ValueError(f"Failed to analyze document structure: {str(e)}")

