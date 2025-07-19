import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState,StateGraph
from prompt_templates.regex_to_e_nfa import regex_to_e_nfa_prompt_template
from prompt_templates.e_nfa_to_dfa import e_nfa_to_dfa_prompt_template
from prompt_templates.dfa_to_minimized_dfa import dfa_to_minimized_dfa_prompt_template
from prompt_templates.push_down_automata import push_down_automata_prompt_template
import uuid



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
        
        
        if ('e_nfa_to_dfa_used' not in st.session_state or st.session_state.e_nfa_to_dfa_used == False) and st.session_state.get('is_pressed_convert', False):
            st.session_state.e_nfa_to_dfa_used = True
            # st.write("debug start..........")
            # st.write(st.session_state.get('input_e_nfa', ''))
            # st.write(st.session_state.get('e_nfa_to_dfa_transition', ''))
            # st.write("debug end..........")
            e_nfa_to_dfa_hint = f"\nHere is the converted DFA transition for the e-NFA {st.session_state.get('latest_input_e_nfa', '')}:\n{st.session_state.get('e_nfa_to_dfa_transition', '')}"
        else:
            e_nfa_to_dfa_hint = ""
        
        
        if ('dfa_to_minimized_dfa_used' not in st.session_state or st.session_state.dfa_to_minimized_dfa_used == False) and st.session_state.get('is_pressed_convert', False):
            st.session_state.dfa_to_minimized_dfa_used = True
            # st.write("debug start..........")
            # st.write(st.session_state.get('input_regex', ''))
            # st.write(st.session_state.get('regex_to_e_nfa_transition', ''))
            # st.write("debug end..........")
            dfa_to_minimized_dfa_hint = f"\nHere is the minimized DFA transition for the given DFA {st.session_state.get('latest_input_dfa', '')}:\n{st.session_state.get('dfa_to_minimized_dfa_transition', '')}"
        else:
            dfa_to_minimized_dfa_hint = ""

        if ('pda_used' not in st.session_state or st.session_state.pda_used == False) and st.session_state.get('is_pressed_convert', False):
            st.session_state.pda_used = True
            # st.write("debug start..........")
            # st.write(st.session_state.get('input_regex', ''))
            # st.write(st.session_state.get('regex_to_e_nfa_transition', ''))
            # st.write("debug end..........")
            push_down_automata_hint = f"\nHere is the converted pda transitions for the given context free language string {st.session_state.get('latest_input_pda', '')}:\n{st.session_state.get('pda_transition', '')}"
        else:
            push_down_automata_hint = ""
        
        
        selected_model = st.session_state.selected_model

        if selected_model['name'] == "Regex-to-ε-NFA":
            prompt = regex_to_e_nfa_prompt_template.invoke({ 
                "messages": state["messages"],
                "regex_to_e_nfa_hint": regex_to_e_nfa_hint
            })

        elif selected_model['name'] == "e_NFA-to-DFA":
            prompt = e_nfa_to_dfa_prompt_template.invoke({ 
                "messages": state["messages"],
                "e_nfa_to_dfa_hint": e_nfa_to_dfa_hint
            })
        elif selected_model['name'] == "PDA":
            prompt = push_down_automata_prompt_template.invoke({ 
                "messages": state["messages"],
                "push_down_automata_hint": push_down_automata_hint
            })
        else:
            prompt = dfa_to_minimized_dfa_prompt_template.invoke({ 
                "messages": state["messages"],
                "dfa_to_minimized_dfa_hint": dfa_to_minimized_dfa_hint
            })



        response = model.invoke(prompt)
        
        if 'is_pressed_convert' in st.session_state:
            st.session_state.is_pressed_convert = False
        return {'messages':response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    st.write()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    return app,config