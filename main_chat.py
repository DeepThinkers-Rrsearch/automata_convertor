import getpass 
import os 
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState,StateGraph


def setup_llm():
    """
    Initialize the llm workflow
    """
    workflow  = StateGraph(state_schema = MessagesState)
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    def call_model(state:MessagesState):
        response = model.invoke(state['messages'])
        return {'messages':response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id":"abc123"}}

    return app,config



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


def main():
    st.set_page_config(
            page_title='LLM Chat Interface with Memory',
            page_icon=':robot_face:',
            layout='wide'
    )

    st.title('LLM Chat Interface with Memory')
    st.markdown("Chat with Gemini 2.0 Flash using LangGraph")

    if 'app' not in st.session_state:
        st.session_state.app,st.session_state.config = setup_llm()

    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation_history(
            st.session_state.app,
            st.session_state.config
        )  

    with st.sidebar:
        st.header('Controls')
        if st.button("Clear Chat History",type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Thread ID:** abc123")
        st.markdown(f"Messages i conversation:** {len(st.session_state.messages)}")

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create human message and invoke the app
                human_message = HumanMessage(content=prompt)
                output = st.session_state.app.invoke(
                    {"messages": [human_message]}, 
                    st.session_state.config
                )
                    
                # Get assistant response
                assistant_response = output["messages"][-1].content
                    
                    # Display assistant response
                st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
            except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()















first_input = [HumanMessage(content="I am chanaka Prasanna")]
first_output = app.invoke({"messages": first_input}, config)
print("Assistant:", first_output["messages"][-1].content)

# 6) Second user turn (memory should kick in)
second_input = [HumanMessage(content="And what about my name now?")]
second_output = app.invoke({"messages": second_input}, config)
print("Assistant:", second_output["messages"][-1].content)

# # 7) Inspect saved conversation (if you want)
# history = memory.get_memory("abc123")
# print("Saved history:", history)