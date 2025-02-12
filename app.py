import streamlit as st
import os
from typing import List, Dict, Optional
import cohere
import json
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4

# Import your RAG components
from retriever import RAGRetriever

@dataclass
class ChatSession:
    """Represents a chat session with metadata."""
    id: str
    name: str
    created_at: str
    messages: List[Dict]
    contexts: List[Dict]



class StreamlitRAGChatbot:
    def __init__(self, retriever, cohere_api_key):
        self.retriever = retriever
        self.co = cohere.Client(cohere_api_key)
    
    def _parse_chunk_structure(self, text: str) -> Dict[str, str]:
        """
        Enhanced parser that better handles investment research structure.
        """
        parsed_data = {}
        current_key = None
        current_value = []
        citation_buffer = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle citations differently
            if line.startswith('[') and line.endswith(']'):
                citation_buffer.append(line)
                continue
                
            if ':' in line and not line.lower().startswith('http'):
                if current_key and current_value:
                    content = ' '.join(current_value).strip()
                    if citation_buffer:
                        content += f"\nCitations: {' '.join(citation_buffer)}"
                    parsed_data[current_key] = content
                    citation_buffer = []
                    
                parts = line.split(':', 1)
                current_key = parts[0].strip()
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:
                if current_key:
                    current_value.append(line)
        
        if current_key and current_value:
            content = ' '.join(current_value).strip()
            if citation_buffer:
                content += f"\nCitations: {' '.join(citation_buffer)}"
            parsed_data[current_key] = content
            
        return parsed_data
    
    def _format_context(self, retrieved_docs: List[Dict], query: str) -> str:
        """
        Enhanced context formatter with better structure preservation.
        """
        context_parts = []
        
        for doc in retrieved_docs:
            parsed_content = self._parse_chunk_structure(doc['text'])
            
            # Start with company/topic header
            context_parts.append(f"\n### {doc['main_heading']}")
            
            # Group related fields for better context
            standard_fields = ['Stage', 'Status', 'Verdict']
            analysis_fields = ['Reasons', 'Keep Watch', 'Did we use brain?']
            
            # Add standard fields first
            for field in standard_fields:
                if field in parsed_content:
                    context_parts.append(f"{field}: {parsed_content[field]}")
            
            # Add analysis fields with proper spacing
            for field in analysis_fields:
                if field in parsed_content:
                    context_parts.append(f"\n{field}:\n{parsed_content[field]}")
            
            # Add any remaining fields
            remaining_fields = set(parsed_content.keys()) - set(standard_fields) - set(analysis_fields)
            for field in remaining_fields:
                context_parts.append(f"\n{field}:\n{parsed_content[field]}")
        
        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str, chat_history: List[Dict]) -> str:
        """
        Investment-specific prompt engineering with better context utilization.
        """
        history_str = self._format_relevant_history(query, chat_history)
        
        return f"""You are an investment research analyst assistant for Golden Gate Ventures, a Singaporean venture capital firm. Your role is to provide precise insights about startups, companies, and investment opportunities based on internal research data.

        Context Rules:
        1. Investment Stage Understanding:
           - When discussing company stages, reference exact details from the data
           - Explain stage-specific implications for investment consideration
           - Connect stage information with status and verdict when relevant
        
        2. Analysis Integration:
           - Combine information from 'Reasons', 'Keep Watch', and 'Did we use brain?' sections
           - Present balanced views incorporating both opportunities and risks
           - Highlight key decision factors from the research
        
        3. Response Requirements:
           - Always ground responses in the provided research data
           - Include relevant citations when present in the source
           - Maintain analytical tone while being clear and concise
           - If data is missing or unclear, acknowledge the gaps
        
        4. Specific Guidelines:
           - For status updates: Explain current position and key factors
           - For investment decisions: Reference specific reasons and verdicts
           - For company analysis: Connect stage, status, and strategic implications
           - For market insights: Link company-specific data to broader trends

        Previous Relevant Discussion:
        {history_str}

        Available Research Data:
        {context}

        Query: {query}

        Provide a structured analysis based on the available research:"""

    def _format_relevant_history(self, query: str, chat_history: List[Dict]) -> str:
        """Enhanced history formatting with investment context awareness."""
        if not chat_history:
            return "No previous discussion."
            
        # Keep last 3 exchanges for context
        recent_history = chat_history[-6:]  
        
        # Check if current query starts new topic
        if len(chat_history) >= 2:
            last_query = chat_history[-2]['content']
            if not self._is_related_query(query, last_query):
                return "New investment topic - no relevant previous discussion."
        
        history_str = "\nRecent relevant discussion:\n"
        for msg in recent_history:
            role = "Analyst" if msg['role'] == "assistant" else "User"
            history_str += f"{role}: {msg['content']}\n"
        
        return history_str

    def _is_related_query(self, current_query: str, previous_query: str) -> bool:
        """Enhanced query relationship detection for investment context."""
        # Convert to sets of words
        current_words = set(current_query.lower().split())
        previous_words = set(previous_query.lower().split())
        
        # Investment-specific keywords that indicate relationship
        investment_terms = {"stage", "status", "investment", "company", "startup", "verdict", "market"}
        
        # Check for shared investment terms
        shared_investment_terms = (current_words.intersection(previous_words)
                                 .intersection(investment_terms))
        
        # Check for company name overlap (assuming capitalized terms might be company names)
        company_names_current = {w for w in current_query.split() if w[0].isupper()}
        company_names_previous = {w for w in previous_query.split() if w[0].isupper()}
        shared_companies = company_names_current.intersection(company_names_previous)
        
        # Consider related if either condition is met
        return len(shared_investment_terms) > 0 or len(shared_companies) > 0

    def get_response(self, query: str, chat_history: List[Dict], max_context_chunks: int = 3):
        """Enhanced response generation with better context management."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=max_context_chunks)
        
        # Format context and create prompt
        context = self._format_context(retrieved_docs, query)
        prompt = self._create_prompt(query, context, chat_history)
        
        # Generate response with tuned parameters
        response = self.co.generate(
            prompt=prompt,
            model='command',
            max_tokens=4096,  # Increased for more comprehensive responses
            temperature=0.3,   # Reduced for more focused responses
            presence_penalty=0.3,
            frequency_penalty=0.3,
        )
        
        return response.generations[0].text.strip(), retrieved_docs
def create_new_chat_session(name: Optional[str] = None) -> ChatSession:
    """Create a new chat session with unique ID."""
    return ChatSession(
        id=str(uuid4()),
        name=name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        created_at=datetime.now().isoformat(),
        messages=[],
        contexts=[]
    )

def initialize_session_state():
    """Initialize session state variables with enhanced chat session management."""
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_chat_id' not in st.session_state:
        # Create initial chat session
        new_session = create_new_chat_session()
        st.session_state.chat_sessions[new_session.id] = new_session
        st.session_state.current_chat_id = new_session.id
    if 'show_context' not in st.session_state:
        st.session_state.show_context = False

def save_chat_sessions():
    """Save all chat sessions to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_sessions_{timestamp}.json"
    
    # Convert sessions to dictionary format
    sessions_data = {
        id: {
            "id": session.id,
            "name": session.name,
            "created_at": session.created_at,
            "messages": session.messages,
            "contexts": session.contexts
        }
        for id, session in st.session_state.chat_sessions.items()
    }
    
    with open(filename, 'w') as f:
        json.dump(sessions_data, f, indent=4)
    return filename

def load_chat_sessions(file):
    """Load chat sessions from a file."""
    content = json.loads(file.read())
    st.session_state.chat_sessions = {
        id: ChatSession(**session_data)
        for id, session_data in content.items()
    }
    if st.session_state.chat_sessions:
        st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions))

def main():
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Investment Research Assistant")
        
        # API Key input
        api_key = st.text_input("Enter Cohere API Key:", type="password")
        if not api_key:
            api_key = os.getenv('COHERE_API_KEY')
        
        # Chat session management
        st.subheader("💬 Chat Sessions")
        
        # Create new chat button
        if st.button("New Chat"):
            new_session = create_new_chat_session()
            st.session_state.chat_sessions[new_session.id] = new_session
            st.session_state.current_chat_id = new_session.id
        
        # Chat session selector
        current_session = st.session_state.chat_sessions[st.session_state.current_chat_id]
        session_names = {
            id: session.name
            for id, session in st.session_state.chat_sessions.items()
        }
        selected_chat = st.selectbox(
            "Select Chat",
            options=list(session_names.keys()),
            format_func=lambda x: session_names[x],
            index=list(session_names.keys()).index(st.session_state.current_chat_id)
        )
        if selected_chat != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_chat
        
        # Rename current chat
        new_name = st.text_input(
            "Rename Current Chat",
            value=current_session.name
        )
        if new_name != current_session.name:
            current_session.name = new_name
        
        # Context display toggle
        st.session_state.show_context = st.toggle(
            "Show Retrieved Context",
            st.session_state.show_context
        )
        
        # Chat history file upload
        st.subheader("💾 Chat Management")
        uploaded_file = st.file_uploader("Load chat sessions", type="json")
        if uploaded_file is not None:
            load_chat_sessions(uploaded_file)
            st.success("Chat sessions loaded!")
        
        # Save chat history
        if st.button("Save All Chats"):
            if st.session_state.chat_sessions:
                filename = save_chat_sessions()
                st.success(f"Chat sessions saved to {filename}")
        
        # Clear current chat
        if st.button("Clear Current Chat"):
            current_session.messages = []
            current_session.contexts = []
            st.success("Current chat cleared!")
    
    # Main chat interface
    st.title(f"💡 {current_session.name}")
    
    # Initialize components
    if api_key:
        retriever = RAGRetriever()
        chatbot = StreamlitRAGChatbot(retriever, api_key)
        
        # Display chat messages
        for i, message in enumerate(current_session.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if (st.session_state.show_context and 
                    message["role"] == "assistant" and 
                    i//2 < len(current_session.contexts)):
                    with st.expander("View Retrieved Context"):
                        st.json(current_session.contexts[i//2])
        
        # Chat input
        if prompt := st.chat_input("Ask about investment opportunities, technologies, or market trends..."):
            # Add user message to chat
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                try:
                    response, retrieved_docs = chatbot.get_response(
                        prompt, 
                        current_session.messages
                    )
                    
                    st.write(response)
                    
                    if st.session_state.show_context:
                        with st.expander("View Retrieved Context"):
                            st.json(retrieved_docs)
                    
                    # Update chat history
                    current_session.messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ])
                    current_session.contexts.append(retrieved_docs)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter your Cohere API key in the sidebar.")

if __name__ == "__main__":
    main()
