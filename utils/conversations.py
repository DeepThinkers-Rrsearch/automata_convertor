import streamlit as st

def load_conversation_history(app,config):
    try:
        current_state =  app.get_state(config)
        if current_state and current_state.get('messages'):
            messages =  current_state['messages']
            history = []


            for msg in messages:
                if hasattr(msg,'content'):
                    if msg.__class__.__name__ == 'HumanMessage':
                        history.append({"role":"user","content":msg.content})
                    else:
                        history.append({"role":"stateforge","content":msg.content})
            
            return history
    except Exception as e:
        st.write(f"No existing history found: {e}")
    return []