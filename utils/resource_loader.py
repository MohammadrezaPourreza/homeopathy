from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import glob

def load_document(file: str) -> list[Document]:
    """
    Load a document from the given file.
    """
    loader = PyPDFLoader(file)
    return loader.load_and_split()


def load_all_documents(base_directory: str) -> list[list[Document]]:
    """
    Load all documents from the given directory.
    """
    documents = []
    for file in glob.glob(base_directory + "/*.pdf"):
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        documents.append(pages)
    return documents

def splitter(text: str, max_length: int = 2000) -> list[Document]:
    """
    Split text into chunks of max_length.
    """
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_length,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )
    return text_splitter.create_documents([text])
