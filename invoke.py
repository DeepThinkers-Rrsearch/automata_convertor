from main_chat import app, config
from langchain_core.messages import HumanMessage

query = "What i said as mt name?"

input_messages = [HumanMessage(query)]



output =  app.invoke({"messages":input_messages},config)
output['messages'][-1].pretty_print()
