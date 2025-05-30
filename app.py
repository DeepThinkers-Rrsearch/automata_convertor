import os
import streamlit as st
import google.generativeai as genai
from utils.dfa_minimization import load_dfa_minimization_model, predict_dfa_minimization
from utils.regex_to_epsilon_nfa import load_regex_to_e_nfa_model,predict_regex_to_e_nfa

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

st.set_page_config(
    page_title='Automata Conversions',
    page_icon=':robot_face:',
    layout='wide'
)

# ─── 1️⃣ Define available models as configuration objects ─────────────────────
models_root = './models'
models = [
    {"name": "DFA-Minimization", "path": os.path.join(models_root, "dfa_minimization")},
    {"name": "Regex-to-ε-NFA", "path": os.path.join(models_root, "regex_to_e_nfa")},
    # {"name": "NFA-to-DFA", "path": os.path.join(models_root, "nfa_to_dfa")},
    # {"name": "PDA", "path": os.path.join(models_root, "pda")},
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

# ─── 2️⃣ Sidebar dropdown with models ──────────────────────────────────────────
model_names = [m["name"] for m in valid_models]
selected_name = st.sidebar.selectbox('Choose Converter', model_names, index=0)

# Get the selected model configuration
selected_model = next(m for m in valid_models if m["name"] == selected_name)


def load_model(model_name: str):

    if model_name == "DFA-Minimization":
        dfa_minimization_model =load_dfa_minimization_model("models/dfa_minimization/dfa_transformer.pt","models/dfa_minimization/dfa_minimizer_tokenizer.pkl")
        return dfa_minimization_model, None, None
    elif model_name == "Regex-to-ε-NFA":
        regex_to_e_nfa_model,stoi, itos = load_regex_to_e_nfa_model("models/regex_to_e_nfa/transformer_regex_to_e_nfa.pt","models/regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl")
        return regex_to_e_nfa_model,stoi, itos

    # Add more model types as needed
    
    return None  # Replace with actual model


# ─── 4️⃣ Main UI ───────────────────────────────────────────────────────────────
st.title("Automata Conversions")

# Display current model info
st.info(f"Selected Model: **{selected_model['name']}**")
st.write(f"Model Path: `{selected_model['path']}`")

# Input area with dynamic placeholder based on selected model
input_placeholder = {
    "DFA-Minimization": "Enter your DFA description (states, transitions, etc.)",
    "Regex-to-ε-NFA": "Enter your regular expression",
    # Add more placeholders for other models
}.get(selected_model['name'], "Enter your input here")

user_input = st.text_area("Input", placeholder=input_placeholder)

if st.button("Convert", type="primary"):
    if not user_input.strip():
        st.warning("Please enter something to convert.")
    else:
        with st.spinner(f"Converting using {selected_model['name']}..."):
            # TODO: Load model (you might want to cache this)
            model,stoi,itos = load_model(selected_model['name'])
            
            result = None
            if selected_model['name'] == "Regex-to-ε-NFA":
                result = predict_regex_to_e_nfa(user_input,model,stoi,itos)
            elif selected_model['name'] == "DFA-Minimization":
                result = predict_dfa_minimization(model,user_input)
 
            
            # Display result
            st.subheader("Conversion Result:")
            st.code(result, language="text")

# ─── 5️⃣ Additional UI sections (optional) ─────────────────────────────────────
with st.expander("Model Information"):
    st.write(f"**Model Name:** {selected_model['name']}")
    st.write(f"**Model Path:** {selected_model['path']}")
    # TODO: Add more model-specific information here

# TODO: Add any model-specific configuration options in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
# TODO: Add model-specific settings/parameters here
