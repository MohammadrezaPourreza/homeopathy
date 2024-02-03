from openai import OpenAI  # noqa: F401
from utils.pinecone import query_index, embed_text
from utils.resource_loader import load_document  # noqa: F401
import streamlit as st
import time


def type_text(text, type_speed=0.02):
    text = text.strip()
    answer_container = st.empty()
    for i in range(len(text) + 1):
        answer_container.markdown(text[0:i])
        time.sleep(type_speed)
    st.divider()

st.set_page_config(
    page_title="Dataherald",
    page_icon="./images/logo.png",
    layout="wide")

st.sidebar.title("Remedy finder")
st.sidebar.write("Ask your question and find the most relevant remedies based on symptoms.")
TOP_K = int(st.sidebar.text_input("Number of remedies to return", 5))

st.title("💬 Homeopathy")
st.caption("🚀 A Tool to find remedies based on the symptoms")

output_container = st.empty()
user_input = st.chat_input("Ask your question")
output_container = output_container.container()
if user_input:
    output_container.chat_message("user").write(user_input)
    answer_container = output_container.chat_message("assistant")

    with st.spinner("Finding Remedies..."):
        prompt_embed = embed_text([user_input])[0]
        result = query_index(prompt_embed, k=TOP_K)
        relevant_remedies = "\n"
        for match in result['matches']:
            remedy_names = match['metadata']['source'].split("/")[-1].replace(".pdf", "")
            relevant_remedies += f"Remedy: {remedy_names} \n, relevance score: {match['score']}\n"
            relevant_remedies += "\n" 

    assistant = f"""
Here are the most relevant remedies for your symptoms: 
{relevant_remedies}
"""
    type_text(assistant)

