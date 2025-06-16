import getpass
import os
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

st.set_page_config(
    page_title='Image data extraction',
    page_icon='...',
    layout='wide'
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

dfa_minimization_extraction_prompt = '''

You are an intelligent agent that extracts DFA (Deterministic Finite Automaton) transitions from a given DFA diagram.

Your task is to analyze the DFA diagram and extract all transitions between states exactly as they are, without missing any information.

Here is how your output format must strictly look like:
"A: a-->A, b-->A; B: a-->B, b-->A; in:A; fi:A"

Explanation of the format:

Uppercase letters (A, B, etc.) denote state names.

For each state, include all transitions in the form: a-->X, where a is the symbol and X is the destination state.

Separate multiple transitions with commas.

Sometimes arrows heads may be overlapped each other. In that case carefully identify the arrow head and extract the transition.

Use semicolons ; to separate different states.

Use in:X to specify the initial state, where X is the state with a long arrow starting from a dot.

Use fi:X,Y,... to list final states (double-circled states). If no final state exists, just write fi:.

If there are multiple final states, list them comma-separated like fi: B,C.

Do not include any explanation or notes in your output—only the exact DFA transition string.

Do not use any newline characters; everything should be returned as one single-line string inside quotes.

Remember: Do not skip any state or transition. Even if a state loops to itself on a symbol, that must be included.

'''

st.title("Image data extraction")
img_input =  st.file_uploader("Upload image of DFA or NFA",type=['png'])

if img_input:
    encoded_image = base64.b64encode(img_input.read()).decode("utf-8")

    message_local = HumanMessage(
        content=[
            {"type": "text", "text": dfa_minimization_extraction_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
        ]
    )


    result_local = llm.invoke([message_local])
    st.write(f"Response for local image: {result_local.content}")
