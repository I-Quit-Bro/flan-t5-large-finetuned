import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import os

# Import the chatbot class
from model import MedicalChatbotRAG

# Path to the checkpoint folder
CHECKPOINT_PATH = "./output"

@st.cache_resource
def load_model():
    """
    Load the fine-tuned model and tokenizer.
    """
    chatbot = MedicalChatbotRAG(checkpoint_path=CHECKPOINT_PATH)
    return chatbot

def main():
    # Page configuration
    st.set_page_config(page_title="Medical Chatbot", layout="centered")

    # Title
    st.title("Medical Chatbot (RAG)")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This is a Retrieval-Augmented Generation (RAG) chatbot fine-tuned for medical question answering. 
        Powered by [Transformers](https://huggingface.co/transformers) and [Streamlit](https://streamlit.io).
        """
    )

    # Chat history container
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm a medical chatbot. How can I assist you today?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(
                f'<div style="text-align: left; background-color: #f1f1f1; padding: 10px; border-radius: 10px; margin-bottom: 10px;">'
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
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Load chatbot and generate response
            chatbot = load_model()
            with st.spinner("Generating response..."):
                response = chatbot.generate_response(user_input)

            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Clear input box
            st.session_state.user_input = ""
        else:
            st.warning("Please type a message before sending.")

if __name__ == "__main__":
    main()
