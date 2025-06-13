import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState,StateGraph
from prompt_templates.regex_to_e_nfa import regex_to_e_nfa_prompt_template

def setup_llm():
    """
    Initialize the llm workflow
    """
    workflow  = StateGraph(state_schema = MessagesState)
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    def call_model(state:MessagesState):
       
        if ('regex_to_e_nfa_used' not in st.session_state or st.session_state.regex_to_e_nfa_used == False) and st.session_state.get('is_pressed_convert', False):
            st.session_state.regex_to_e_nfa_used = True
            # st.write("debug start..........")
            # st.write(st.session_state.get('input_regex', ''))
            # st.write(st.session_state.get('regex_to_e_nfa_transition', ''))
            # st.write("debug end..........")
            regex_to_e_nfa_hint = f"\nHere is the converted ε-NFA transition for the regular expression {st.session_state.get('latest_input_regex', '')}:\n{st.session_state.get('regex_to_e_nfa_transition', '')}"
        else:
            regex_to_e_nfa_hint = ""
        
        prompt = regex_to_e_nfa_prompt_template.invoke({ 
            "messages": state["messages"],
            "regex_to_e_nfa_hint": regex_to_e_nfa_hint
            
            })
        
        response = model.invoke(prompt)
        if 'is_pressed_convert' in st.session_state:
            st.session_state.is_pressed_convert = False
        return {'messages':response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id":"abc123"}}

    return app,config