from typing import Dict, List, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass, field

@dataclass
class ContextStack:
    document_title: str = ""
    current_section_id: str = "root"
    section_path: List[str] = field(default_factory=list)  # Section IDs from root to current
    heading_path: List[str] = field(default_factory=list)  # Section titles from root to current

    # List context
    list_stack: List[Dict[str, Any]] = field(default_factory=list)
    current_list_type: Optional[str] = None
    current_list_level: int = 0

    # Table context
    in_table: bool = False
    table_headers: List[str] = field(default_factory=list)
    current_row_index: int = -1
    current_cell_index: int = -1

    # Other context
    in_code_block: bool = False
    in_blockquote: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> 'ContextStack':
        """Create a copy of the current context stack"""
        return ContextStack(
            document_title=self.document_title,
            current_section_id=self.current_section_id,
            section_path=self.section_path.copy(),
            heading_path=self.heading_path.copy(),
            list_stack=[item.copy() for item in self.list_stack],
            current_list_type=self.current_list_type,
            current_list_level=self.current_list_level,
            in_table=self.in_table,
            table_headers=self.table_headers.copy(),
            current_row_index=self.current_row_index,
            current_cell_index=self.current_cell_index,
            in_code_block=self.in_code_block,
            in_blockquote=self.in_blockquote,
            metadata=self.metadata.copy()
        )

def get_full_context_string(self) -> str:
    """Generate a string representation of the current context"""
    context_parts = []
    
    # Add document title
    context_parts.append(f"Document: {self.document_title}")
    
    # Add heading path
    if self.heading_path:
        context_parts.append(f"Path: {' > '.join(self.heading_path)}")
    
    # Add list context if applicable
    if self.current_list_level > 0:
        list_type = self.current_list_type or "list"
        context_parts.append(f"In {list_type} (level {self.current_list_level})")
    
    # Add table context if applicable
    if self.in_table:
        context_parts.append(f"In table (row {self.current_row_index + 1}, cell {self.current_cell_index + 1})")
        if self.table_headers:
            current_header = self.table_headers[self.current_cell_index] if 0 <= self.current_cell_index < len(self.table_headers) else None
            if current_header:
                context_parts.append(f"Column: {current_header}")
    
    # Add code block context
    if self.in_code_block:
        context_parts.append("In code block")
        
    # Add blockquote context
    if self.in_blockquote:
        context_parts.append("In blockquote")
    
    return " | ".join(context_parts)

def initialize_context_stack() -> ContextStack:
    """
    Initialize the context stack for document processing.

    Returns:
        A new ContextStack object initialized with default values
    """
    return ContextStack(
        document_title="",
        current_section_id="root",
        section_path=["root"],
        heading_path=[],
        list_stack=[],
        current_list_type=None,
        current_list_level=0,
        in_table=False,
        table_headers=[],
        current_row_index=-1,
        current_cell_index=-1,
        in_code_block=False,
        in_blockquote=False,
        metadata={}
    )

