import streamlit as st
import wikipedia
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Fusion AI", page_icon="🤖")

st.title("🤖 Fusion AI")
st.markdown("A personal chatbot that helps you chat, feel, and code!")

# Load model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.past_inputs.append(user_input)

    # Tokenize input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.markdown(f"**Fusion AI:** {response}")
