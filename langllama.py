from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.llms.langchain import LangChainLLM
from langchain_together import Together
from langchain.chat_models import ChatOpenAI
import logging
from pathlib import Path

# Step 1: Load and Process Documents
def load_documents(document_path):
    reader = SimpleDirectoryReader(input_dir=document_path)
    documents = reader.load_data()
    return documents

# Step 2: Initialize Sentence Splitter for Chunking
def chunk_documents(documents, chunk_size=1024, chunk_overlap=200):
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
    return nodes

# Step 3: Define Embedding Model
def setup_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Step 4: Define LLM Model
def setup_llm(api_key):
    together_llm = Together(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        temperature=0.7,
        together_api_key=api_key
    )
    llm = LangChainLLM(llm=together_llm)
    return llm

# Step 5: Configure Settings for LLM and Embedding Model
def setup_settings(llm, embed_model):
    # Update Settings globally instead of creating a new instance
    Settings.llm = llm
    Settings.embed_model = embed_model
    return Settings

# Step 6: Create Vector Store Index from Documents
def create_vector_store_index(documents, nodes, storage_dir="./storage_mini"):
    vector_index = VectorStoreIndex.from_documents(
        documents, 
        show_progress=True, 
        node_parser=nodes
    )
    vector_index.storage_context.persist(persist_dir=storage_dir)
    return vector_index

# Step 7: Load Vector Store Index
def load_vector_store_index(storage_dir):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = VectorStoreIndex.load_index_from_storage(storage_context)
    return index

# Step 8: Define Query Engine
def create_query_engine(index):
    query_engine = index.as_query_engine()
    return query_engine

# Step 9: Execute Query
def execute_query(query_engine, query):
    response = query_engine.query(query)
    print(response.response)

# Main Function
def main():
    document_path = "/Users/likhithaparuchuri/projects/llamaImpact/docs"
    api_key = "fbfb3084148fa76a6dcfb97852ef52321779645e9ad32206aaf4e638e3ca406b"

    # Load and process documents
    documents = load_documents(document_path)
    nodes = chunk_documents(documents)

    # Initialize models and settings
    embed_model = setup_embedding_model()
    llm = setup_llm(api_key)
    settings = setup_settings(llm, embed_model)

    # Create vector store index
    vector_index = create_vector_store_index(documents, nodes)

    # Create query engine
    query_engine = create_query_engine(vector_index)

    # Execute a query
    user_query = "Explain MoE?"
    execute_query(query_engine, user_query)

if __name__ == "__main__":
    main()