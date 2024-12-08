import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import os

# class import
from model import MedicalChatbotRAG

# path
CHECKPOINT_PATH = "paste your path here"

@st.cache_resource
def load_model():

    chatbot = MedicalChatbotRAG(checkpoint_path=CHECKPOINT_PATH)
    return chatbot

def main():
    st.set_page_config(page_title="Medical Chatbot", layout="centered")

    st.title("Medical Chatbot (RAG)")

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This is a Retrieval-Augmented Generation (RAG) chatbot fine-tuned for medical question answering. 
        Powered by [Transformers](https://huggingface.co/transformers) and [Streamlit](https://streamlit.io).
        """
    )
    #chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm a medical chatbot. How can I assist you today?"}
        ]

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(
                f'<div style="text-align: left; background-color: #e0e0e0; padding: 10px; border-radius: 10px; margin-bottom: 10px;">'
                f'<b>Assistant:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align: right; background-color: #dff9fb; padding: 10px; border-radius: 10px; margin-bottom: 10px;">'
                f'<b>You:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )

    # User input
    user_input = st.text_input("Type your message here:", key="user_input", placeholder="Enter your query...")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})

            chatbot = load_model()
            with st.spinner("Generating response..."):
                response = chatbot.generate_response(user_input)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.user_input = ""
        else:
            st.warning("Please type a message before sending.")

if __name__ == "__main__":
    main()
