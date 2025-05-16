import streamlit as st
import json # For attempting to parse and display JSON nicely

# Attempt to import from the existing script.
# We might need to adjust interactive_rfp_agent.py to make components more easily importable
# or ensure it doesn't run its own main loop when imported.
from interactive_rfp_agent import (
    RFPQueryEngine,
    format_results_for_display,
    get_final_llm_response,
    SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE,
    SCAFFOLDING_PLACEHOLDER,
    FINAL_LLM_SYSTEM_PROMPT,
    PINECONE_HOST as DEFAULT_PINECONE_HOST, # Import default for convenience
    # Note: interactive_rfp_agent.py still has global PINECONE_API_KEY, OPENAI_API_KEY
    # We will override these with Streamlit inputs / secrets.
)

# --- Page Configuration ---
st.set_page_config(page_title="Interactive RFP Agent", layout="wide")

# --- API Key Management ---
st.sidebar.title("API Configuration")
st.sidebar.info(
    "For deployed apps, use Streamlit Secrets (e.g., create a .streamlit/secrets.toml file).\n"
    "PINECONE_API_KEY = \"your_pinecone_key\"\n"
    "PINECONE_HOST = \"your_pinecone_host\"\n"
    "OPENAI_API_KEY = \"your_openai_key\"\n\n"
    "For local development, you can enter them here. Values from secrets will be used if available."
)

# Try to get secrets first, then allow override via text input
pinecone_api_key_val = st.secrets.get("PINECONE_API_KEY", "")
pinecone_host_val = st.secrets.get("PINECONE_HOST", DEFAULT_PINECONE_HOST)
openai_api_key_val = st.secrets.get("OPENAI_API_KEY", "")

pinecone_api_key_input = st.sidebar.text_input(
    "Pinecone API Key", 
    type="password", 
    value=pinecone_api_key_val,
    help="Your Pinecone API key."
)
pinecone_host_input = st.sidebar.text_input(
    "Pinecone Host URL", 
    value=pinecone_host_val,
    help="Your Pinecone index host URL."
)
openai_api_key_input = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    value=openai_api_key_val,
    help="Your OpenAI API key."
)

# --- Main App ---
st.title("Interactive RFP Agent")
st.caption("Answering your RFP questions using Pinecone and OpenAI.")

# Initialize RFPQueryEngine
engine = None
if pinecone_api_key_input and pinecone_host_input:
    try:
        engine = RFPQueryEngine(
            pinecone_host=pinecone_host_input,
            pinecone_api_key=pinecone_api_key_input,
            openai_api_key=openai_api_key_input if openai_api_key_input else None
        )
    except Exception as e:
        st.error(f"Failed to initialize RFP Query Engine: {e}")
        st.stop()
else:
    st.warning("Please provide Pinecone API Key and Host URL in the sidebar to start.")
    st.stop()

# Initialize chat history and first query flag in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_first_query" not in st.session_state:
    st.session_state.is_first_query = True

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message["role"] == "assistant":
            if "pinecone_results_str" in message:
                with st.expander("View Pinecone Search Results", expanded=False):
                    st.markdown(message["pinecone_results_str"])
            
            if "final_json_response_str" in message and message["final_json_response_str"]:
                st.markdown("##### Synthesized JSON Response")
                try:
                    # Attempt to parse and display as formatted JSON
                    json_data = json.loads(message["final_json_response_str"])
                    st.json(json_data)
                except json.JSONDecodeError:
                    # If not valid JSON, display as code block
                    st.code(message["final_json_response_str"], language="text")
            elif "error_message" in message:
                 st.error(message["error_message"])


# Get user input
if prompt := st.chat_input("Ask about Totogi's capabilities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant's turn
    with st.chat_message("assistant"):
        assistant_response_message = {}
        
        raw_user_query_for_qa = prompt
        query_for_general_searches = prompt

        if st.session_state.is_first_query:
            query_for_general_searches = SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE.replace(SCAFFOLDING_PLACEHOLDER, prompt)
            st.info(f"Applying initial context scaffold for Pinecone searches (this happens only on the first query).")
            st.session_state.is_first_query = False

        pinecone_results_str = ""
        final_json_str = ""
        error_msg = None

        try:
            with st.spinner("Searching Pinecone..."):
                all_search_results = engine.search(query_for_general_searches, raw_user_query_for_qa)
                pinecone_results_str = format_results_for_display(all_search_results)
            
            assistant_response_message["pinecone_results_str"] = pinecone_results_str
            with st.expander("View Pinecone Search Results (Current Query)", expanded=True):
                st.markdown(pinecone_results_str)

            if openai_api_key_input:
                with st.spinner("Synthesizing final JSON response with OpenAI..."):
                    pinecone_context_str = ""
                    for search_name, data_items in all_search_results.items():
                        pinecone_context_str += f"\nContext from '{search_name}':\n"
                        if data_items:
                            for i, data_item in enumerate(data_items, 1):
                                content = getattr(data_item, 'text', "")
                                metadata = getattr(data_item, 'metadata', {})
                                # Try to find a meaningful source identifier
                                source = metadata.get('source') or metadata.get('filename') or metadata.get('title') or search_name
                                pinecone_context_str += f"  Result {i} (source: {source}): {content}\n"
                        else:
                            pinecone_context_str += "  No results found.\n"

                    final_json_str = get_final_llm_response(
                        FINAL_LLM_SYSTEM_PROMPT,
                        prompt, # Original user prompt for this turn's final synthesis
                        pinecone_context_str,
                        openai_api_key_input
                    )
                
                assistant_response_message["final_json_response_str"] = final_json_str
                st.markdown("##### Synthesized JSON Response")
                try:
                    json_data = json.loads(final_json_str)
                    st.json(json_data)
                except json.JSONDecodeError:
                    st.code(final_json_str, language="text") # Display as raw text if not valid JSON
            else:
                st.warning("OpenAI API Key not provided in sidebar. Skipping final JSON synthesis.")
                final_json_str = "OpenAI API Key not provided. Cannot generate final JSON response."
                assistant_response_message["final_json_response_str"] = final_json_str


        except Exception as e:
            st.error(f"An error occurred: {e}")
            error_msg = f"An error occurred during processing: {e}"
            assistant_response_message["error_message"] = error_msg
        
        st.session_state.messages.append({
            "role": "assistant", 
            **assistant_response_message
        })
        # Streamlit automatically re-runs from top to bottom on widget interaction,
        # so messages will be displayed. 