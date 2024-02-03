from utils.pinecone import create_index, upsert_documents
from utils.resource_loader import load_all_documents


if "__main__" == __name__:
    docs = load_all_documents("./resources/MateriaMedica-John Henry Clarke/")
    index = create_index()
    texts = []
    sources = []
    for doc in docs:
        content = ""
        for page in doc:
            content += page.page_content
        content = content.replace("\n", " ")
        texts.append(content)
        sources.append(doc[0].metadata["source"])
    upsert_documents(index, texts, sources)