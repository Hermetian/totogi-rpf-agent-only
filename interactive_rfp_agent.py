from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Type
import json # Added import

import requests
import openai # type: ignore
from langchain_core.vectorstores import VectorStore # type: ignore
from langchain.schema import Document # type: ignore

# Langflow specific imports - assuming these are available in the environment
# If not, these would need to be stubbed or replaced with local definitions.
try:
    from langflow.schema import Data
    from langflow.helpers.data import docs_to_data
    from langflow.base.vectorstores.model import LCVectorStoreComponent
    from langflow.io import (
        StrInput,
        SecretStrInput,
        IntInput,
        HandleInput,
    )
except ImportError:
    print("Warning: Langflow specific imports failed. Some features might not work as expected.")
    print("Please ensure langflow is installed if you see errors related to Data, docs_to_data, etc.")
    # Define a simple Data class if langflow is not available for basic functionality
    from dataclasses import dataclass, field
    @dataclass
    class Data:
        text: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    def docs_to_data(docs: List[Document]) -> List[Data]:
        return [Data(text=doc.page_content, metadata=doc.metadata) for doc in docs]

    class LCVectorStoreComponent: # Minimal stub
        def __init__(self, *args, **kwargs):
            self.embedding = None
            self.search_type = "Similarity"
            self.search_kwargs: Dict[str, Any] = {}
            self.search_query: Optional[str] = None
            # inputs for LCVectorStoreComponent often include these
            self.inputs: List[Any] = [] # Placeholder

# --- Configuration ---
# Replace with your actual API keys and host, or use environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://yowling-wobster-d6glwko.svc.aped-4627-b74a.pinecone.io") # e.g., "https://your-index-xxxx.svc.gcp-starter.pinecone.io"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

DEFAULT_INDEX_NAME = "yowling-wobster" # Default index name from pinecone_search.py
DEFAULT_TEXT_KEY = "chunk_text" # Default text_key from pinecone_search.py

HYDE_PROMPT = """You are HyDE, a retrieval helper that fabricates a *plausible* answer to any user query so that the answer can be embedded and used for vector search.  **Rules**  1. Reply with a short, coherent answer (≈ 1–3 sentences) that a knowledgeable source *might* give. 2. Pack the reply with concrete facts, entities, dates, and domain-specific vocabulary that are likely to appear in relevant documents. 3. Do **not** mention that the answer is hypothetical, do **not** add caveats, citations, or meta-commentary, and do **not** reference this instruction set. 4. If the query is ambiguous, choose the most common interpretation and answer accordingly. 5. Return *only* the answer text—no bullets, headings, or extra formatting.  Begin."""

SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE = "You are an expert telecommunications consultant evaluating Totogi's capabilities. Your task is to evaluate if Totogi can meet the given requirement: <original prompt text should go here>"
SCAFFOLDING_PLACEHOLDER = "<original prompt text should go here>"

FINAL_LLM_SYSTEM_PROMPT = """You are an expert telecommunications consultant. Based on the user's requirement and the provided supporting information, your task is to evaluate if Totogi can meet the requirement.
Your response MUST be in valid JSON format with the following structure:
{
  "compliance": "Fully Compliant" OR "Partially Compliant" OR "Not Compliant",
  "explanation": "A brief explanation of how Totogi can or cannot support the requirement. Be concise yet comprehensive.",
  "sources": [
    // List 1 to 4 unique source identifiers, ordered from most to least relevant.
    // Each identifier MUST combine the 'Search Scenario Name' (e.g., "Default Namespace", "QA-Pairs Namespace (Original Query)") 
    // and its 'Result number' (e.g., 1, 2, 3) from the supporting information.
    // Format: "Search Scenario Name #". For example: "Default Namespace 3", "QA-Pairs Namespace (Original Query) 1".
    // Only include a source if it directly supports the explanation.
    // If fewer than 4 sources are highly relevant, or if some results are significantly more informative than others, list only the most relevant ones.
  ]
}"""

# --- Adapted from bss_magic_generic/pinecone_search.py ---

class PineconeHTTPVectorStore(VectorStore):
    """Read-only wrapper around Pinecone's Serverless /records HTTP API."""

    def __init__(self, *, host: str, namespace: str, api_key: str, text_key: str):
        self.host = host.rstrip("/")
        self.namespace = namespace.strip()
        self.api_key = api_key
        self.text_key = text_key

    def similarity_search(self, query: str, k: int = 4, **_) -> List[Document]:
        url = f"{self.host}/records/namespaces/{self.namespace}/search"
        # The Pinecone /records/search API expects a vector if not using a text query with a model.
        # The original pinecone_search.py example implies it handles text queries, possibly via an embedding model
        # configured elsewhere or a Pinecone feature that embeds text queries.
        # For this direct HTTP wrapper, if Pinecone serverless expects vectors, this part needs an embedding step.
        # However, the original code payload was: {"query": {"inputs": {"text": query}, "top_k": k}}
        # This suggests Pinecone might handle text embedding server-side for some configurations.
        # Let's assume the payload structure from the original script is correct.
        payload = {"query": {"inputs": {"text": query}, "top_k": k}}
        headers = {"Api-Key": self.api_key, "Content-Type": "application/json"}
        
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30) # Increased timeout
            res.raise_for_status()
            response_json = res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during Pinecone API call to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                try:
                    print(f"Response content: {e.response.json()}")
                except ValueError:
                    print(f"Response content: {e.response.text}")
            return []


        hits = response_json.get("result", {}).get("hits", [])
        documents = []
        for hit in hits:
            fields = hit.get("fields", {})
            content = None
            possible_keys = [self.text_key, "chunk_text", "text", "content", "meta_preview"]
            # Add any other common keys you expect
            
            for possible_key in possible_keys:
                if possible_key in fields and fields[possible_key]:
                    content = fields[possible_key]
                    break
            
            if not content: # Fallback if specific keys not found
                for key, value in fields.items():
                    if isinstance(value, str) and value.strip() and len(value) > 20:
                        content = value
                        break
            
            if not content:
                content = f"No primary text content found. Available fields: {', '.join(fields.keys())}"

            documents.append(Document(page_content=str(content), metadata=fields))
        return documents

    def add_texts(self, *_a, **_kw): # type: ignore
        raise NotImplementedError("PineconeHTTPVectorStore is read-only.")

    @classmethod
    def from_texts(cls: Type[PineconeHTTPVectorStore], texts: List[str], embedding: Any, ids: Optional[List[str]] = None, **kwargs: Any) -> PineconeHTTPVectorStore: # type: ignore
        raise NotImplementedError(
            "PineconeHTTPVectorStore is read-only; use a full Pinecone client "
            "if you need to build an index from local texts."
        )

class PineconeDirectQueryComponent(LCVectorStoreComponent): # type: ignore
    """
    Adapted component to search a Pinecone serverless index, 
    with optional OpenAI query rewrite.
    This is a simplified version for direct use, not a full Langflow component.
    """
    display_name = "Pinecone Direct Query (Adapted)"
    
    # Mimicking Langflow input definitions for clarity on what's needed
    # These will be set as attributes directly in this script.
    index_name: str = DEFAULT_INDEX_NAME
    host: str = PINECONE_HOST
    namespace: str = ""
    pinecone_api_key: str = PINECONE_API_KEY
    text_key: str = DEFAULT_TEXT_KEY
    openai_api_key: Optional[str] = OPENAI_API_KEY
    openai_prompt: Optional[str] = None
    query: Optional[str] = "" # User's query, can be set directly
    search_query: Optional[str] = "" # Also for user's query, for compatibility
    number_of_results: int = 4
    
    # For LCVectorStoreComponent compatibility if its methods are called
    embedding: Any = None 
    search_type: str = "Similarity"
    search_kwargs: Dict[str,Any] = {}


    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs) # type: ignore
        # Initialize attributes from kwargs or defaults
        self.index_name = kwargs.get("index_name", DEFAULT_INDEX_NAME)
        self.host = kwargs.get("host", PINECONE_HOST)
        self.namespace = kwargs.get("namespace", "")
        self.pinecone_api_key = kwargs.get("pinecone_api_key", PINECONE_API_KEY)
        self.text_key = kwargs.get("text_key", DEFAULT_TEXT_KEY)
        self.openai_api_key = kwargs.get("openai_api_key", OPENAI_API_KEY)
        self.openai_prompt = kwargs.get("openai_prompt", None)
        self.query = kwargs.get("query", "")
        self.search_query = kwargs.get("search_query", "")
        self.number_of_results = kwargs.get("number_of_results", 4)
        self.status: Any = None # For compatibility with original, though not used by UI here

    def build_vector_store(self) -> VectorStore:
        if not self.host or not self.pinecone_api_key:
            raise ValueError("Pinecone host and API key are required.")
        return PineconeHTTPVectorStore(
            host=self.host,
            namespace=self.namespace,
            api_key=self.pinecone_api_key,
            text_key=self.text_key,
        )

    def _transform_query(self, query_text: str) -> str:
        if not (self.openai_api_key and (self.openai_prompt or "").strip()):
            # print(f"DEBUG: OpenAI rewrite disabled - missing API key or prompt for query: '{query_text}'")
            return query_text

        # print(f"DEBUG: Attempting OpenAI rewrite for query: '{query_text}'")
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Consider making model configurable
                messages=[
                    {"role": "system", "content": self.openai_prompt},
                    {"role": "user", "content": query_text},
                ],
                max_tokens=128,
                temperature=0.75,
            )
            rewritten_query = response.choices[0].message.content # type: ignore
            rewritten_query = rewritten_query.strip() if rewritten_query else query_text
            # print(f"DEBUG: Original query: '{query_text}'")
            # print(f"DEBUG: Rewritten query: '{rewritten_query}'")
            return rewritten_query
        except Exception as exc:
            error_msg = f"OpenAI rewrite error: {exc} (falling back to original query)"
            print(f"DEBUG: {error_msg}")
            self.status = error_msg # Set status for potential logging
            return query_text

    def search_documents(self) -> List[Data]: # Type hint should be List[Data]
        query_text = (self.query or self.search_query or "").strip()
        if not query_text:
            return []

        effective_query = self._transform_query(query_text)
        
        query_status = (
            f"Query transformation: '{query_text}' → '{effective_query}'" 
            if query_text != effective_query else 
            "Using original query (no transformation applied)"
        )
        # print(f"DEBUG: {query_status}")

        try:
            vector_store = self.build_vector_store()
            docs = vector_store.similarity_search(
                query=effective_query,
                k=self.number_of_results,
            )
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
        
        # print(f"DEBUG: Found {len(docs)} documents for query '{effective_query}'")

        enhanced_data: List[Data] = []
        for doc in docs:
            # Using simpler metadata structure for direct use
            metadata_payload = {
                **doc.metadata, # Original metadata from Pinecone hit
                "document_content": doc.page_content, # Ensure content is in metadata for Langflow Data pattern
                "original_query": query_text,
                "effective_query": effective_query,
                "query_info": query_status
            }
            # Ensure page_content is string
            page_content_str = str(doc.page_content) if doc.page_content is not None else ""
            
            # Using docs_to_data helper if Langflow is available, otherwise manual Data creation
            try:
                # Create a Langchain Document first for docs_to_data
                enhanced_doc = Document(page_content=page_content_str, metadata=metadata_payload)
                data_item = docs_to_data([enhanced_doc])[0]
            except NameError: # docs_to_data not defined (Langflow import failed)
                 data_item = Data(text=page_content_str, metadata=metadata_payload) # type: ignore
            
            enhanced_data.append(data_item)
        
        self.status = enhanced_data # Store for potential inspection
        return enhanced_data

# --- RFP Query Engine ---
class RFPQueryEngine:
    def __init__(self, pinecone_host: str, pinecone_api_key: str, openai_api_key: Optional[str]):
        self.pinecone_host = pinecone_host
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key

        if not self.pinecone_host or self.pinecone_host == "YOUR_PINECONE_HOST_URL":
            print("Warning: PINECONE_HOST is not configured. Searches will likely fail.")
        if not self.pinecone_api_key or self.pinecone_api_key == "YOUR_PINECONE_API_KEY":
            print("Warning: PINECONE_API_KEY is not configured. Searches will likely fail.")
        if not self.openai_api_key or self.openai_api_key == "YOUR_OPENAI_API_KEY":
            print("Warning: OPENAI_API_KEY is not configured for HyDE search. HyDE will be disabled.")


    def _create_search_component(self, namespace: str, index_name: str = DEFAULT_INDEX_NAME, 
                                 hyde_prompt: Optional[str] = None, text_key: str = DEFAULT_TEXT_KEY,
                                 num_results: int = 4) -> PineconeDirectQueryComponent:
        
        current_openai_key = self.openai_api_key if hyde_prompt else None
        current_openai_prompt = hyde_prompt if hyde_prompt else None

        return PineconeDirectQueryComponent(
            host=self.pinecone_host,
            pinecone_api_key=self.pinecone_api_key,
            openai_api_key=current_openai_key,
            namespace=namespace,
            index_name=index_name,
            text_key=text_key,
            openai_prompt=current_openai_prompt,
            number_of_results=num_results
        )

    def search(self, primary_query: str, raw_user_query: str) -> Dict[str, List[Data]]:
        results: Dict[str, List[Data]] = {}

        search_scenarios = [
            {"name": "Default Namespace", "namespace": "", "hyde_prompt": None, "num_results": 4, "use_primary_query": True},
            {"name": "OneBill Namespace", "namespace": "onebill", "hyde_prompt": None, "num_results": 4, "use_primary_query": True},
            {"name": "QA-Pairs Namespace (HyDE Query)", "namespace": "qa-pairs", "hyde_prompt": HYDE_PROMPT, "num_results": 4, "use_primary_query": True},
            {"name": "QA-Pairs Namespace (Original Query)", "namespace": "qa-pairs", "hyde_prompt": None, "num_results": 4, "use_primary_query": False},
        ]

        for scenario in search_scenarios:
            print(f"Performing search in '{scenario['name']}'...")
            component = self._create_search_component(
                namespace=scenario["namespace"],
                hyde_prompt=scenario["hyde_prompt"],
                num_results=scenario.get("num_results", 4)
            )
            
            if scenario.get("use_primary_query", True):
                component.query = primary_query
            else:
                component.query = raw_user_query
            
            # Check if HyDE is attempted without API key
            if scenario["hyde_prompt"] and (not self.openai_api_key or self.openai_api_key == "YOUR_OPENAI_API_KEY"):
                print(f"Skipping HyDE for '{scenario['name']}' as OpenAI API key is not configured.")
                search_results = []
            else:
                search_results = component.search_documents()
            
            results[scenario["name"]] = search_results
            # print(f"Found {len(search_results)} results for '{scenario['name']}'.")
        
        return results

# --- Output Formatting, Final LLM Call & Main Loop ---

def format_results_for_display(all_results: Dict[str, List[Data]]) -> str:
    display_str = ""
    for search_name, data_items in all_results.items():
        display_str += f"\n--- Results from: {search_name} ---\n"
        if not data_items:
            display_str += "No results found.\n"
            continue
        for i, data_item in enumerate(data_items, 1):
            content = getattr(data_item, 'text', "[Content not found]")
            metadata = getattr(data_item, 'metadata', {})
            
            display_str += f"Result {i}:\n"
            display_str += f"  Content: {content}\n"
            
            # Display key metadata if available
            if metadata:
                display_str += "  Metadata:\n"
                if "original_query" in metadata:
                    display_str += f"    Original Query: {metadata['original_query']}\n"
                if "effective_query" in metadata and metadata.get("original_query") != metadata.get("effective_query"):
                    display_str += f"    Effective (HyDE) Query: {metadata['effective_query']}\n"
                
                # Display some source information if present in various forms
                source_info_keys = ["source", "filename", "title", "document_name"]
                for src_key in source_info_keys:
                    if src_key in metadata and metadata[src_key]:
                        display_str += f"    {src_key.capitalize()}: {metadata[src_key]}\n"
                        break # Show first available source info
                
                # Optionally, display a few other metadata fields
                other_meta_to_show = {k: v for k, v in metadata.items() if k not in ['original_query', 'effective_query', 'document_content', 'query_info'] and k not in source_info_keys and isinstance(v, (str, int, float))}
                if other_meta_to_show:
                     for k, v in list(other_meta_to_show.items())[:2]: # Show max 2 other fields
                         display_str += f"    {k.capitalize()}: {v}\n"
            display_str += "\n"
    return display_str

def get_final_llm_response(system_prompt: str, user_requirement: str, pinecone_context: str, openai_api_key: Optional[str]) -> str:
    if not openai_api_key or openai_api_key == "YOUR_OPENAI_API_KEY":
        return "OpenAI API Key not configured. Cannot generate final LLM response."

    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        full_user_message = f"""User's Requirement:
{user_requirement}

---
Supporting Information from Pinecone Searches:
{pinecone_context}"""
        
        # print(f"DEBUG: Final LLM System Prompt:\n{system_prompt}")
        # print(f"DEBUG: Final LLM User Message:\n{full_user_message[:500]}...") # Print start of user message

        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o as a powerful model, can be changed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_message},
            ],
            temperature=0.2, # Lower temperature for more deterministic JSON output
            # response_format={"type": "json_object"} # Enable if using newer models that support this
        )
        final_response = response.choices[0].message.content
        return final_response.strip() if final_response else "LLM returned an empty response."
    except Exception as e:
        print(f"Error during final LLM call: {e}")
        return f"Error generating final LLM response: {e}"

def main_chat_loop():
    # Ensure API keys are set, otherwise print a strong warning.
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY" or \
       PINECONE_HOST == "YOUR_PINECONE_HOST_URL" :
        print("\n" + "="*60)
        print("!!! CRITICAL WARNING !!!")
        print("Pinecone API Key or Host is not configured in the script.")
        print("Please set PINECONE_API_KEY and PINECONE_HOST variables.")
        print("Searches will likely fail until these are set.")
        print("="*60 + "\n")
    
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("\n" + "="*60)
        print("!!! OPENAI WARNING !!!")
        print("OpenAI API Key is not configured. HyDE search will be disabled.")
        print("Set OPENAI_API_KEY variable if you want to use HyDE.")
        print("="*60 + "\n")


    engine = RFPQueryEngine(PINECONE_HOST, PINECONE_API_KEY, OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY" else None)
    
    print("Welcome to the Interactive RFP Query Agent!")
    print("The agent will search three Pinecone configurations for each query.")
    print("It will retain conversation context. Type 'new' to reset context and start a new query session.")
    print("Type 'exit' or 'quit' to end the conversation.")

    is_first_query = True
    current_conversation_context = "" # Stores the history of the current conversation session

    while True:
        try:
            user_input = input("\nYour query: ").strip()
        except EOFError: # Handle Ctrl+D
            print("\nExiting...")
            break
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        if user_input.lower() == 'new':
            current_conversation_context = ""
            is_first_query = True # Reset for scaffolding on the next actual query
            print("\nContext has been reset. Starting a new query session.")
            continue

        if not user_input:
            continue

        print("\nProcessing your query...")
        
        # This is the text passed to the final LLM as the "user's requirement"
        # It includes past context and the current user input.
        user_requirement_for_llm = (
            f"{current_conversation_context}\n\nUser: {user_input}"
            if current_conversation_context
            else f"User: {user_input}" # Start with "User:" for clarity even on the first turn
        )

        # This is the query for "QA-Pairs Namespace (Original Query)"
        # It should reflect the full conversational context.
        raw_user_query_for_qa_search = user_requirement_for_llm

        # This is the primary query for general searches and HyDE.
        if is_first_query: # True for the first query after start or after "new"
            # Apply scaffolding only to the current user_input for the very first query of a session
            primary_query_for_engine = SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE.replace(SCAFFOLDING_PLACEHOLDER, user_input)
            print(f"DEBUG: Initial query (scaffolded for general/HyDE): \n{primary_query_for_engine[:300]}...")
        else:
            # For subsequent queries in a conversation, HyDE and general searches operate on the full context.
            primary_query_for_engine = user_requirement_for_llm
            
        all_search_results = engine.search(primary_query_for_engine, raw_user_query_for_qa_search)
        formatted_output = format_results_for_display(all_search_results)
        
        print("\n--- Agent Response (Retrieved Pinecone Documents) ---")
        print(formatted_output)

        # --- Add Final LLM Call for JSON synthesis ---
        print("\nSynthesizing final JSON response with GPT-4.x...")
        
        pinecone_context_str = ""
        for search_name, data_items in all_search_results.items():
            pinecone_context_str += f"\nContext from '{search_name}':\n"
            if data_items:
                for i, data_item in enumerate(data_items, 1):
                    content = getattr(data_item, 'text', "")
                    metadata = getattr(data_item, 'metadata', {})
                    source = metadata.get('source') or metadata.get('filename') or search_name
                    pinecone_context_str += f"  Result {i} (source: {source}): {content}\n"
            else:
                pinecone_context_str += "  No results found.\n"
        
        final_json_response = get_final_llm_response(
            FINAL_LLM_SYSTEM_PROMPT,
            user_requirement_for_llm, # Pass the full contextual requirement
            pinecone_context_str,
            OPENAI_API_KEY if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY" else None
        )
        
        print("\n--- Synthesized JSON Response (GPT-4.x) ---")
        print(final_json_response)

        # Update conversation history
        agent_response_summary = "Could not parse agent's explanation from JSON." # Default
        try:
            response_data = json.loads(final_json_response)
            if isinstance(response_data, dict) and "explanation" in response_data:
                agent_response_summary = response_data["explanation"]
            elif isinstance(response_data, str): # If LLM failed to produce valid JSON
                 agent_response_summary = final_json_response 
        except json.JSONDecodeError:
            # If JSON parsing fails, use the raw response if it's short, otherwise a placeholder
            if len(final_json_response) < 250: # Avoid polluting history with very long non-JSON strings
                agent_response_summary = final_json_response
            else:
                agent_response_summary = "Agent provided a detailed response (non-JSON format)."
        except Exception: # Catch any other unexpected errors during parsing
            agent_response_summary = "Error parsing agent's response for history."


        current_conversation_context += f"\n\nUser: {user_input}\nAgent: {agent_response_summary}"
        is_first_query = False # Set to False after the first query in a session is processed

# if __name__ == "__main__":
#     main_chat_loop() 