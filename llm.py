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