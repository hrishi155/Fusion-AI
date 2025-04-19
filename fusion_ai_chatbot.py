# fusion_ai_app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

st.set_page_config(page_title="Fusion AI", page_icon="🤖", layout="centered")

st.title("🤖 Fusion AI - Your LLM Assistant")
st.markdown("Ask anything — from coding help to current affairs, general queries, or emotional support.")

@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-2.7B"  # ✅ Open-access model without the need for login
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

llm = load_model()

# Chat interface
user_prompt = st.text_area("💬 You:", height=100, placeholder="Ask Fusion AI anything...")

if st.button("Generate Response"):
    if user_prompt.strip() != "":
        with st.spinner("Fusion AI is thinking..."):
            response = llm(user_prompt)[0]["generated_text"]
            # Remove prompt from start of generated text if it repeats
            if response.startswith(user_prompt):
                response = response[len(user_prompt):]
            st.markdown(f"**🧠 Fusion AI:** {response.strip()}")
    else:
        st.warning("Please enter a question.")
