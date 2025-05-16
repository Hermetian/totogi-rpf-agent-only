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
st.caption("Answering your RFP questions. Type 'new' to reset. First query gets JSON, follow-ups are conversational. Prefix with 'search ' for new Pinecone search.")

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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_first_query" not in st.session_state: # True for the first query of a session
    st.session_state.is_first_query = True
if "last_pinecone_context" not in st.session_state:
    st.session_state.last_pinecone_context = ""
if "llm_conversation_history" not in st.session_state:
    st.session_state.llm_conversation_history = ""

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message["role"] == "assistant":
            if message.get("type") == "info": 
                st.info(message["content"])
            else:
                if "pinecone_results_str" in message and message["pinecone_results_str"]:
                    with st.expander("View Pinecone Search Results", expanded=False):
                        st.markdown(message["pinecone_results_str"])
                
                # Adaptive display of LLM response based on whether JSON was expected
                if message.get("is_first_query_response"): # Check if this response was for a first query
                    st.markdown("##### Synthesized JSON Response")
                    if "final_llm_output" in message and message["final_llm_output"]:
                        try:
                            json_data = json.loads(message["final_llm_output"])
                            st.json(json_data)
                        except json.JSONDecodeError:
                            st.code(message["final_llm_output"], language="text")
                elif "final_llm_output" in message and message["final_llm_output"]:
                    st.markdown("##### Agent's Response")
                    st.markdown(message["final_llm_output"]) # Display as markdown for conversational follow-ups
                
                if "error_message" in message: # Ensure errors are always shown
                    st.error(message["error_message"])


# Get user input
if prompt := st.chat_input("Ask about Totogi... (Type 'new' to reset; 'search <query>' to force search)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if prompt.strip().lower() == "new":
        st.session_state.is_first_query = True
        st.session_state.last_pinecone_context = ""
        st.session_state.llm_conversation_history = ""
        with st.chat_message("assistant"):
            st.info("Context reset. First query will expect JSON. Ready for a new query session.")
        st.session_state.messages.append({"role": "assistant", "content": "Context reset. First query will expect JSON. Ready for a new query session.", "type": "info"})
    
    else: 
        assistant_response_message = {}
        perform_search_this_turn = False
        query_for_general_searches = "" 
        raw_user_query_for_qa = ""      
        
        # Capture if this turn is considered the "first query" for response formatting purposes
        expect_json_response_this_turn = st.session_state.is_first_query
        assistant_response_message["is_first_query_response"] = expect_json_response_this_turn

        if st.session_state.is_first_query:
            perform_search_this_turn = True
            st.info("Applying initial context scaffold for Pinecone searches (first query of session).")
            query_for_general_searches = SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE.replace(SCAFFOLDING_PLACEHOLDER, prompt)
            raw_user_query_for_qa = prompt 
        elif prompt.lower().startswith("search "): 
            perform_search_this_turn = True
            actual_search_term = prompt[len("search "):].strip()
            if not actual_search_term:
                with st.chat_message("assistant"):
                    st.warning("Please provide a query after 'search '.")
                st.session_state.messages.append({"role": "assistant", "content": "Please provide a query after 'search '.", "type": "info"})
                st.stop() 
            
            st.info(f"Performing new Pinecone search for: '{actual_search_term}'")
            query_for_general_searches = actual_search_term 
            raw_user_query_for_qa = actual_search_term      
        else: 
            st.info("Using existing Pinecone context. To force a new search, prefix your query with 'search '.")

        if perform_search_this_turn:
            try:
                with st.spinner("Searching Pinecone..."):
                    all_search_results = engine.search(query_for_general_searches, raw_user_query_for_qa)
                    current_pinecone_results_for_display = format_results_for_display(all_search_results)
                    new_pinecone_context_for_llm = ""
                    for search_name, data_items in all_search_results.items():
                        new_pinecone_context_for_llm += f"\nContext from '{search_name}':\n"
                        if data_items:
                            for i, data_item in enumerate(data_items, 1):
                                content = getattr(data_item, 'text', "")
                                metadata = getattr(data_item, 'metadata', {})
                                source = metadata.get('source') or metadata.get('filename') or metadata.get('title') or search_name
                                new_pinecone_context_for_llm += f"  Result {i} (source: {source}): {content}\n"
                        else:
                            new_pinecone_context_for_llm += "  No results found.\n"
                    st.session_state.last_pinecone_context = new_pinecone_context_for_llm
                    assistant_response_message["pinecone_results_str"] = current_pinecone_results_for_display
                    # No need to display Pinecone results immediately here if it's done in the history loop
                    # However, for user experience, showing it in an expander for the current turn is good.
                    # This expander will be shown *before* the LLM response for the current turn.
                    if current_pinecone_results_for_display: # Only show if there are results
                        with st.expander("View Pinecone Search Results (Current Query)", expanded=True):
                            st.markdown(current_pinecone_results_for_display)
                    
                    # CRITICAL: Set is_first_query to False *after* a successful search for the first query has completed.
                    if st.session_state.is_first_query: 
                        st.session_state.is_first_query = False

            except Exception as e:
                st.error(f"Error during Pinecone search: {e}")
                assistant_response_message["error_message"] = f"Error during Pinecone search: {e}"
                st.session_state.messages.append({"role": "assistant", **assistant_response_message})
                st.stop() 

        llm_input_user_requirement = st.session_state.llm_conversation_history
        if llm_input_user_requirement:
            llm_input_user_requirement += "\n\n"
        llm_input_user_requirement += f"User: {prompt}"

        llm_output_str = ""
        with st.chat_message("assistant"):
            if openai_api_key_input:
                if not st.session_state.last_pinecone_context and not perform_search_this_turn:
                    st.warning("No Pinecone context from previous searches is available. LLM response may be general.")
                
                current_llm_system_prompt = FINAL_LLM_SYSTEM_PROMPT if expect_json_response_this_turn else ""

                with st.spinner("Synthesizing final response with OpenAI..."):
                    llm_output_str = get_final_llm_response(
                        current_llm_system_prompt,
                        llm_input_user_requirement, 
                        st.session_state.last_pinecone_context, 
                        openai_api_key_input
                    )
                
                assistant_response_message["final_llm_output"] = llm_output_str
                
                # Adaptive display for the current turn being processed
                if expect_json_response_this_turn:
                    st.markdown("##### Synthesized JSON Response") 
                    try:
                        json_data = json.loads(llm_output_str)
                        st.json(json_data)
                    except json.JSONDecodeError:
                        st.code(llm_output_str, language="text")
                else:
                    st.markdown("##### Agent's Response") 
                    st.markdown(llm_output_str)

            else:
                st.warning("OpenAI API Key not provided. Skipping final synthesis.")
                llm_output_str = "OpenAI API Key not provided. Cannot generate final response."
                assistant_response_message["final_llm_output"] = llm_output_str
                # Display this warning directly
                if expect_json_response_this_turn:
                    st.markdown("##### Synthesized JSON Response") 
                    st.code(llm_output_str, language="text")
                else:
                    st.markdown("##### Agent's Response") 
                    st.markdown(llm_output_str)

        # Update LLM conversation history
        agent_explanation_for_history = "LLM response not processed or explanation unavailable."
        if llm_output_str:
            if expect_json_response_this_turn:
                try:
                    parsed_json = json.loads(llm_output_str)
                    if isinstance(parsed_json, dict):
                        agent_explanation_for_history = parsed_json.get("explanation", "No explanation in JSON.")
                    else: 
                        agent_explanation_for_history = str(parsed_json) 
                except json.JSONDecodeError:
                    agent_explanation_for_history = llm_output_str # Use raw string if not JSON
            else: # For conversational follow-ups, the whole output is the explanation
                agent_explanation_for_history = llm_output_str
        
        st.session_state.llm_conversation_history = llm_input_user_requirement + f"\nAgent: {agent_explanation_for_history}"
        st.session_state.messages.append({"role": "assistant", **assistant_response_message})
        # Streamlit reruns, redrawing messages