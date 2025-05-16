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
# API keys are now exclusively managed by Streamlit Secrets for deployed apps.
# Ensure PINECONE_API_KEY, PINECONE_HOST, and OPENAI_API_KEY are set in Streamlit Cloud dashboard.

pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pinecone_host = st.secrets.get("PINECONE_HOST", DEFAULT_PINECONE_HOST) # Allow default host if not in secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# --- Main App ---
st.title("Interactive RFP Agent")
st.caption("Answering your RFP questions. Type 'new' to reset. First query gets JSON, follow-ups are conversational. Prefix with 'search ' for new Pinecone search.")

# Initialize RFPQueryEngine
engine = None
if pinecone_api_key and pinecone_host:
    try:
        engine = RFPQueryEngine(
            pinecone_host=pinecone_host,
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key if openai_api_key else None
        )
    except Exception as e:
        st.error(f"Failed to initialize RFP Query Engine: {e}")
        st.stop()
else:
    st.error("Required API secrets (PINECONE_API_KEY, PINECONE_HOST) not found. "
             "Please configure them in your Streamlit Cloud app settings.")
    st.sidebar.warning("Ensure PINECONE_API_KEY and PINECONE_HOST are set in your Streamlit app secrets.")
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
        actual_search_term = "" # Initialize actual_search_term
        
        # Capture if this turn is considered the "first query" for response formatting purposes
        expect_json_response_this_turn = st.session_state.is_first_query
        assistant_response_message["is_first_query_response"] = expect_json_response_this_turn

        if st.session_state.is_first_query:
            perform_search_this_turn = True
            # Info message about scaffolding is now part of the assistant's turn display below
            query_for_general_searches = SCAFFOLDING_PROMPT_FOR_PINECONE_TEMPLATE.replace(SCAFFOLDING_PLACEHOLDER, prompt)
            raw_user_query_for_qa = prompt 
        elif prompt.lower().startswith("search "): 
            perform_search_this_turn = True
            actual_search_term = prompt[len("search "):].strip()
            if not actual_search_term:
                # This warning will be displayed within the upcoming assistant message block
                pass # Handled in the with st.chat_message("assistant") block
            else:
                query_for_general_searches = actual_search_term 
                raw_user_query_for_qa = actual_search_term      
        else: 
            # No st.info here for "Using existing context", it will be part of assistant's message if no search

        # Perform operations within the assistant's chat message context where appropriate
        with st.chat_message("assistant"):
            # Display informational messages based on search type for this turn
            if st.session_state.is_first_query and perform_search_this_turn:
                st.info("Applying initial context scaffold for Pinecone searches (first query of session).")
            elif prompt.lower().startswith("search ") and perform_search_this_turn:
                if not actual_search_term: # Handle empty search term case here
                    st.warning("Please provide a query after 'search '.")
                    # Store this warning in the message history and stop further processing for this turn.
                    assistant_response_message["type"] = "info" # Mark as an info/warning message
                    assistant_response_message["final_llm_output"] = "Please provide a query after 'search ' to initiate a new search."
                    st.session_state.messages.append({"role": "assistant", **assistant_response_message})
                    st.stop() # Stop processing for this turn
                else:
                    st.info(f"Performing new Pinecone search for: '{actual_search_term}'")
            elif not perform_search_this_turn and not (prompt.strip().lower() == "new") : # Avoid double message for "new"
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
                        
                        if current_pinecone_results_for_display:
                            with st.expander("View Pinecone Search Results (Current Query)", expanded=True):
                                st.markdown(current_pinecone_results_for_display)
                        
                        if st.session_state.is_first_query: 
                            st.session_state.is_first_query = False

                except Exception as e:
                    st.error(f"Error during Pinecone search: {e}")
                    assistant_response_message["error_message"] = f"Error during Pinecone search: {e}"
                    # No need to append to messages here, it will be done at the end of the main 'else' block
                    # However, we do need to store the error in the assistant_response_message for history
                    st.session_state.messages.append({"role": "assistant", **assistant_response_message})
                    st.stop() 

            llm_input_user_requirement = st.session_state.llm_conversation_history
            if llm_input_user_requirement:
                llm_input_user_requirement += "\n\n"
            llm_input_user_requirement += f"User: {prompt}"

            llm_output_str = ""
            if openai_api_key: # Check if OpenAI key is available from secrets
                if not st.session_state.last_pinecone_context and not perform_search_this_turn:
                    st.warning("No Pinecone context from previous searches is available. LLM response may be general.")
                
                current_llm_system_prompt = FINAL_LLM_SYSTEM_PROMPT if expect_json_response_this_turn else ""

                with st.spinner("Synthesizing final response with OpenAI..."):
                    llm_output_str = get_final_llm_response(
                        current_llm_system_prompt,
                        llm_input_user_requirement, 
                        st.session_state.last_pinecone_context, 
                        openai_api_key # Use directly fetched secret
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

            else: # OpenAI API Key not found in secrets
                st.warning("OpenAI API Key not found in secrets. Skipping final synthesis.")
                llm_output_str = "OpenAI API Key not found. Cannot generate final response."
                assistant_response_message["final_llm_output"] = llm_output_str
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
                        agent_explanation_for_history = llm_output_str
                else: # For conversational follow-ups, the whole output is the explanation
                    agent_explanation_for_history = llm_output_str
            
            st.session_state.llm_conversation_history = llm_input_user_requirement + f"\nAgent: {agent_explanation_for_history}"
            # Append the complete assistant message (which now includes UI elements directly printed in the with block)
            # to history *after* the with st.chat_message("assistant") block has completed.
            # The actual UI elements (st.info, st.spinner, st.markdown, st.json, st.code, st.expander) are rendered 
            # when they are called. Adding to st.session_state.messages ensures they are part of the scrollback history correctly.
            # We only need to add the *data* that the history rendering loop uses.
            
        # This should be outside the `with st.chat_message("assistant")` block if that block only handles immediate display
        st.session_state.messages.append({"role": "assistant", **assistant_response_message})
        # Streamlit reruns, redrawing messages