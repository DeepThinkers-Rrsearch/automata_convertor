import os
import streamlit as st
from utils.dfa_minimization import load_dfa_minimization_model, predict_dfa_minimization,load_tokenizer
from utils.regex_to_epsilon_nfa import load_regex_to_e_nfa_model,predict_regex_to_e_nfa
from utils.e_nfa_to_dfa import load_e_nfa_to_dfa_model,predict_e_nfa_to_dfa
from utils.push_down_automata import load_PDA_model,predict_PDA_transitions
from utils.graphviz.graphviz_regex_to_e_nfa import epsilon_nfa_to_dot
from utils.graphviz.graphviz_minimized_dfa import minimized_dfa_to_dot
from utils.graphviz.graphviz_dfa import dfa_output_to_dot
from utils.graphviz.graphviz_pda import pda_output_to_dot
from utils.llm import setup_llm
from utils.conversations import load_conversation_history
from langchain_core.messages import HumanMessage
from utils.classes.regex_conversion_stack import RegexConversionStack



st.set_page_config(
    page_title='State Forge',
    page_icon='⚙️',
    layout='wide'
)



dfa_minimization_extraction_prompt = '''

Here in this process you are a agent that extract an DFA transitons from the given input. You need to extract each and every relationships/transitions as it is. You should not miss any information. This is the sample outpust shouls looks like.

"A: a-->A, b-->A; B: a-->B, b-->A; in:A; fi:A"

Outpus is an string. Here Capital letters denotes the State names. Letter in double rounded circle is the final state. If any state has a arrow started with a dot, that is the initial state. 

In above example, "in" denotes the initial states, "fi" represent final states.

If there is more than one final states, those should be denoted using commas. As a example, "fi: D,E".

Hint: for initial state, you can see a long arrow started with a dot.

Also no preembles in the output. Just the required output string


'''


models_root = './models'
models = [
    {"name": "DFA-Minimization", "path": os.path.join(models_root, "dfa_minimization")},
    {"name": "Regex-to-ε-NFA", "path": os.path.join(models_root, "regex_to_e_nfa")},
    {"name": "e_NFA-to-DFA", "path": os.path.join(models_root, "e_nfa_to_dfa")},
    {"name": "PDA", "path": os.path.join(models_root, "pda")},
]

# Validate that model paths exist
valid_models = []
for model_config in models:
    if os.path.isdir(model_config["path"]):
        valid_models.append(model_config)
    else:
        st.warning(f"Model path not found: {model_config['path']}")

if not valid_models:
    st.error("No valid models available.")
    st.stop()

model_names = [m["name"] for m in valid_models]
selected_name = st.sidebar.selectbox('Choose Converter', model_names, index=0)


selected_model = next(m for m in valid_models if m["name"] == selected_name)

if selected_model['name'] == "Regex-to-ε-NFA":
    if "regex_stack" not in st.session_state:
        st.session_state.regex_stack = RegexConversionStack()


def load_model(model_name: str):

    if model_name == "DFA-Minimization":
        dfa_minimization_model =load_dfa_minimization_model("models/dfa_minimization/dfa_minimizer_transformer.pt","models/dfa_minimization/dfa_minimizer_tokenizer.pkl")
        return dfa_minimization_model, None, None
    elif model_name == "Regex-to-ε-NFA":
        regex_to_e_nfa_model,stoi, itos = load_regex_to_e_nfa_model("models/regex_to_e_nfa/transformer_regex_to_e_nfa.pt","models/regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl")
        return regex_to_e_nfa_model,stoi, itos
    elif model_name == "e_NFA-to-DFA":
        e_nfa_to_dfa_model = load_e_nfa_to_dfa_model("models/e_nfa_to_dfa/transformer_model.pt")
        return e_nfa_to_dfa_model, None, None
    elif model_name == "PDA":
        pda_model = load_PDA_model("models/pda/pda.pth")
        return pda_model, None, None

    return None  # Replace with actual model


def clear_on_convert():
    if st.session_state.conversion_result:
        st.session_state.conversion_result = None
        st.session_state.conversion_graph = None
        st.session_state.diagram_png_bytes = None
        st.session_state.latest_input_regex = None
        st.session_state.regex_to_e_nfa_transition = None
        st.session_state.regex_to_e_nfa_used  = False


st.session_state.pressed_once = False

# Input area with dynamic placeholder based on selected model
input_placeholder = {
    "DFA-Minimization": "Enter your DFA description (states, transitions, etc.)",
    "Regex-to-ε-NFA": "Enter your regular expression",
    "PDA": "Enter your language example string...\nEg:- aabb (a^nb^n)"
    # Add more placeholders for other models
}.get(selected_model['name'], "Enter your input here")

input_img_bytes = None
img_input = None

if selected_model['name'] == "DFA-Minimization" or selected_model['name'] == "NFA-to-DFA":
    img_input =  st.file_uploader("Upload image of DFA or NFA",type=['png','jpg','jpeg','svg'])
    

user_input = st.text_area("Input", placeholder=input_placeholder)

if selected_model['name'] == "Regex-to-ε-NFA":
    st.session_state.latest_input_regex = user_input


if st.button("Convert", type="primary"):
    if not user_input.strip():
        st.warning("Please enter something to convert.")
    else:
        with st.spinner(f"Converting using {selected_model['name']}..."):
            
            model,stoi,itos = load_model(selected_model['name'])
            
            result = None
            graph =  None
            png_bytes = None

            if selected_model['name'] == "Regex-to-ε-NFA":
                result = predict_regex_to_e_nfa(user_input,model,stoi,itos)
                st.session_state.regex_to_e_nfa_transition = result
                st.session_state.regex_stack.push(user_input,result)
                st.session_state.is_pressed_convert = True
                if "regex_to_e_nfa_used" in st.session_state: 
                    st.session_state.regex_to_e_nfa_used = False
                graph =epsilon_nfa_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "DFA-Minimization":
                result = predict_dfa_minimization(model,user_input)
                graph = minimized_dfa_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "e_NFA-to-DFA":
                result = predict_e_nfa_to_dfa(model,user_input)
                graph =dfa_output_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "PDA":
                result = predict_PDA_transitions(model,user_input)
                graph =pda_output_to_dot(result)
                png_bytes = graph.pipe(format="png")
            
            st.session_state.conversion_result = result
            st.session_state.conversion_graph  = graph
            st.session_state.diagram_png_bytes = png_bytes



if 'conversion_result' in st.session_state and "diagram_png_bytes" in st.session_state:
    st.subheader("Conversion Result:")
    st.code(st.session_state.conversion_result, language="text")
    st.subheader("Generated Diagram:")
    st.graphviz_chart(st.session_state.conversion_graph.source)

    if st.session_state.diagram_png_bytes:
        st.subheader("Download Diagram as PNG")
        st.download_button(
            label="⬇️ Download (PNG)",
            data=st.session_state.diagram_png_bytes,
            file_name="diagram.png",
            mime="image/png"
        )
            

if selected_model['name'] == "Regex-to-ε-NFA":
    if 'app' not in st.session_state:
        st.session_state.app,st.session_state.config = setup_llm()

    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation_history(
            st.session_state.app,
            st.session_state.config
        )
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

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(f"Selected Model: **{selected_model['name']}**")
st.sidebar.write(f"Model Path: `{selected_model['path']}`")

st.sidebar.markdown("---")
st.sidebar.subheader('Controls')
if st.sidebar.button("Clear Chat History",type="secondary"):
    if selected_model['name'] == "Regex-to-ε-NFA":
        st.session_state.messages = []
        raise st.experimental_rerun()
if st.sidebar.button("View your conversion history"):
    @st.dialog("Conversion History")
    def conversion_history():
        st.write("Conversion History")
        history = st.session_state.regex_stack.all_items()
        if not history:
            st.info("No conversions found yet.")
            return
        
        for idx, item in enumerate(history[::-1], start=1):  # Show most recent first
            st.markdown(f"### 🔢 Conversion {idx}")
            st.markdown(f"**Regex:** `{item['regex']}`")
            st.markdown("**Conversion Result:**")
            st.code(item['conversion'], language='text')
            st.markdown("---")
    conversion_history()


st.sidebar.markdown("---")
if selected_model['name'] == "Regex-to-ε-NFA":
    st.sidebar.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")

