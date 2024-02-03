from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm
from load_dotenv import load_dotenv

import time

load_dotenv()

pc = Pinecone()   
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)

def create_index():
    index_name = 'homeopathy'
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]
    if index_name not in existing_indexes:
        pc.create_index(
            index_name,
            dimension=3072,
            metric='cosine',
            spec=spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    return pc.Index(index_name)

def query_index(question_embedding, k=5):
    index = pc.Index('homeopathy')
    result = index.query(
        vector=question_embedding,
        top_k=k,
        include_metadata=True,
        return_vector=True
    )
    return result

def embed_text(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        end_idx = min(i+batch_size, len(texts))
        embeddings.extend(embed_model.embed_documents(texts[i:end_idx]))
    return embeddings

def delete_index():
    pc.delete_index('homeopathy')

def upsert_documents(index, texts, sources, batch_size=100):
    print(f"number of texts: {len(texts)}")
    for i in tqdm(range(0, len(texts), batch_size)):
        i_end = min(len(texts), i+batch_size)
        batch = texts[i:i_end]
        sources_batch = sources[i:i_end]
        ids = [str(i) for i in range(i, i_end)]
        embeds = embed_text(batch)
        metadata = [{"source": source} for source in sources_batch]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))