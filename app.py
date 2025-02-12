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
        Dynamically parses any structured content into key-value pairs.
        This method replaces the previous rigid structure parsing.
        """
        parsed_data = {}
        current_key = None
        current_value = []
        
        # Split by newlines to handle multi-line content
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a heading (contains ':')
            if ':' in line:
                # Save previous key-value pair if exists
                if current_key and current_value:
                    parsed_data[current_key] = ' '.join(current_value).strip()
                    
                # Split new heading into key and potential value
                parts = line.split(':', 1)
                current_key = parts[0].strip()
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:
                # Add to current value if we have a key
                if current_key:
                    current_value.append(line)
        
        # Save final key-value pair
        if current_key and current_value:
            parsed_data[current_key] = ' '.join(current_value).strip()
        
        return parsed_data
    
    def _format_context(self, retrieved_docs: List[Dict], query: str) -> str:
        """
        Formats retrieved documents into context string, handling any structure.
        """
        context_parts = []
        
        for doc in retrieved_docs:
            # Parse the document structure
            parsed_content = self._parse_chunk_structure(doc['text'])
            
            # Start with the main heading
            context_parts.append(f"\nTopic: {doc['main_heading']}")
            
            # Add all parsed fields, maintaining their original structure
            for key, value in parsed_content.items():
                context_parts.append(f"{key}: {value}")
        
        return "\n".join(context_parts)
    

    def _create_prompt(self, query: str, context: str, chat_history: List[Dict]) -> str:
        """
        Creates a prompt that handles varying chunk structures and maintains context.
        """
        # Format recent relevant chat history
        history_str = self._format_relevant_history(query, chat_history)
        
        return f"""You are an internal efficiency assistant helping team members understand various aspects of projects, investments, and operations. Your role is to provide clear, accurate information based on the available data.

        Guidelines for your response:
        1. Information Structure:
           - Each topic may have different fields and information types
           - Pay attention to the specific fields present in the relevant chunks
           - Don't assume any fixed structure - analyze what's actually present
        
        2. Response Approach:
           - If asked about specific fields, focus on those exactly as they appear
           - For general questions, present relevant information in a logical order
           - Explain connections between different fields when helpful
           - When discussing decisions or statuses, include the supporting context
        
        3. Formatting Guidelines:
           - Present information clearly and logically
           - Use the exact same terminology found in the data
           - If discussing multiple topics, clearly separate them
           - Maintain the original structure while making it readable
        
        Previous Relevant Conversation:
        {history_str}

        Available Information:
        {context}

        Query: {query}

        Please provide a clear, well-structured response based on the available information:"""

    def _format_relevant_history(self, query: str, chat_history: List[Dict]) -> str:
        """Format relevant chat history based on query context."""
        if not chat_history:
            return "No previous conversation."
            
        recent_history = chat_history[-4:]  # Keep last 2 exchanges
        
        # Check if the current query is starting a new topic
        if len(chat_history) >= 2:
            last_query = chat_history[-2]['content']
            if not self._is_related_query(query, last_query):
                return "New topic started - no relevant previous conversation."
        
        history_str = ""
        for msg in recent_history:
            history_str += f"{msg['role'].title()}: {msg['content']}\n"
        
        return history_str

    def _is_related_query(self, current_query: str, previous_query: str) -> bool:
        """Determine if queries are related based on content similarity."""
        current_words = set(current_query.lower().split())
        previous_words = set(previous_query.lower().split())
        overlap = len(current_words.intersection(previous_words))
        return overlap >= 2

    def get_response(self, query: str, chat_history: List[Dict], max_context_chunks: int = 3):
        """
        Generates response for the query with flexible data handling.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=max_context_chunks)
        
        # Format context and create prompt
        context = self._format_context(retrieved_docs, query)
        prompt = self._create_prompt(query, context, chat_history)
        
        # Generate response with parameters tuned for consistency
        response = self.co.generate(
            prompt=prompt,
            model='command',
            max_tokens=800,
            temperature=0.4,  # Lower temperature for more consistent responses
            presence_penalty=0.2,
            frequency_penalty=0.2,
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