from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def create_chroma_retrieval(chroma_db, model_name="gpt-4o", temperature=0.3, k=4):
    # Creates a RetrievalQA chain that uses ChromaDB as a retriever and OpenAI as the LLM.
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=512,
    )
    
    # Turn ChromaDB into a retriever
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",   
        return_source_documents=True
    )
    return qa_chain

def create_faiss_retrieval(faiss_db, model_name="gpt-4o", temperature=0.3, k=4):
    # Creates a RetrievalQA chain that uses ChromaDB as a retriever and OpenAI as the LLM.
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=512,
    )
    
    # Turn ChromaDB into a retriever
    retriever = faiss_db.as_retriever(search_kwargs={"k": k})
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",   # Simple concatenation of chunks into one prompt
        return_source_documents=True  # Optional: to inspect which chunks were used
    )
    
    return qa_chain
