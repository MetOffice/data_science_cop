from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os

def create_chroma_vectorstore(chunks, directory, model):
    # Initialize OpenAI embedding model
    embedding_model = OpenAIEmbeddings(model=model)
    persist_directory=os.path.join(directory, 'chromadb')
    # Create and persist Chroma vector store
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    # Persist the database to disk
    vectordb.persist()
    print(f"Saved {len(chunks)} embeddings to ChromaDB at '{persist_directory}'")
    return None

def create_faiss_vectorstore(chunks, directory, model):
    # Initialize the OpenAI embedding model
    embeddings_model = OpenAIEmbeddings(model=model)
    persist_directory=os.path.join(directory, 'faissdb')
    # Create the FAISS vector store from texts
    vectordb = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings_model
    )
    # Persist FAISS index to disk
    vectordb.save_local(persist_directory)
    print(f"Saved {len(chunks)} embeddings to FAISS at '{persist_directory}'")
    return None 

