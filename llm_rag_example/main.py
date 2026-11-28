from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
from utils import load_website, clean_text, chunk_text
from vectorstore import create_chroma_vectorstore, create_faiss_vectorstore
from retrievers import create_chroma_retrieval, create_faiss_retrieval

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = "add-your-openapi-key"
    url = "https://cylc.github.io/cylc-doc/stable/html/index.html"
    embedding_model="text-embedding-3-large"
    vectorstore_parent_dir="add-storage-path"

    data = load_website(url, max_pages=10)
    # to load large website in batch and async
    #data = load_website_in_batch(url, max_pages=3000, batch_size=50, max_concurrent=20)

    print(f"\nLoaded {len(data)} documents total.")
    content = data[0].page_content

    # Clean text with unwanted tags
    cleaned = clean_text(content)
    
    # Create chunks
    chunks = chunk_text(
        cleaned, 
        chunk_size=1200, 
        overlap=200
        )

    # create and save embeddings to chromaDB
    create_chroma_vectorstore(
        chunks, 
        vectorstore_parent_dir,
        embedding_model
        )

    # create and save embeddings to FAISS
    create_faiss_vectorstore(
        chunks,
        vectorstore_parent_dir,
        embedding_model
        )

    ######### Now create retrievers and invoke with query

    model = OpenAIEmbeddings(model=embedding_model)
    query = "Write any basic example of cylc8 workflow?"

    # Create Chromadb retrever and run a query
    chroma_vectorstore = "chromadb"
    persist_directory = os.path.join(vectorstore_parent_dir, chroma_vectorstore)
    chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=model
    )

    qa_chain = create_chroma_retrieval(
        chroma_db, 
        model_name="gpt-4o", 
        temperature=0.2
        )
    response = qa_chain.invoke({"query": query})

    print("\nQuestion:", query)
    print("\nAnswer:", response["result"])

    # Create Chromadb retrever and run a query
    faiss_vectorstore = "faissdb"
    persist_directory = os.path.join(vectorstore_parent_dir, faiss_vectorstore)
    
    faiss_db = FAISS.load_local(
        persist_directory, 
        model, 
        allow_dangerous_deserialization=True
        )
    
    qa_chain = create_faiss_retrieval(
        faiss_db, 
        model_name="gpt-4o", 
        temperature=0.2
        )
    response = qa_chain.invoke({"query": query})

    print("\nQuestion:", query)
    print("\nAnswer:", response["result"])


