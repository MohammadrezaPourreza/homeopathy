from utils.pinecone import create_index, upsert_documents, delete_index
from utils.resource_loader import load_all_documents, splitter
from load_dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage
)
from langchain.chat_models import ChatOpenAI
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

RED = "\033[91m"  # Red
YELLOW = "\033[93m"  # Yellow
RESET = "\033[0m"  # Reset text color
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_field = "text"
chat = ChatOpenAI(
    model='gpt-3.5-turbo-1106'
)

messages = [
    SystemMessage(content="You are a helpful assistant. Expert in homeopathy."),
]

def augment_prompt(query: str, vectorstore: Pinecone, k:int = 5) -> str:
    results = vectorstore.similarity_search(query, k=k)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

if __name__ == "__main__":
    
    pages = load_all_documents("./resources")
    full_text = ""
    for page in pages:
        for p in page:
            full_text += p.page_content
    docs = splitter(full_text)
    texts = []
    for doc in docs:
        texts.append(doc.page_content)
    index = create_index()
    upsert_documents(index, texts)
    vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
    )
    query = input(RED + "User: " + RESET)
    while query != "quit":
        prompt = HumanMessage(
            content=augment_prompt(query, vectorstore)
        )
        messages.append(prompt)

        res = chat(messages)
        
        # Display Assistant's response in yellow
        assistant_response = YELLOW + "Assistant: " + res.content + RESET
        print(assistant_response)

        messages.append(res)
        query = input(RED + "User: " + RESET)
    delete_index()