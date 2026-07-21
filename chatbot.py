# chatbot.py
from openai import OpenAI  # Keep for future use
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd
import ollama
from ollama import embed
import os
import time
import ui

load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")  # Uncomment for future OpenAI use

# Load facts
df = pd.read_csv("space_facts.csv")

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient('./db/myfactsdb')
collection_name = "MySpaceFacts"

# Precompute embeddings once
def generate_embeddings(embedding_name):
    # if embedding_name=='OpenAI Embedding':
    #     embedding = embedding_functions.OpenAIEmbeddingFunction(
    #         api_key=openai_key,
    #         model_name="text-embedding-3-small"
    #     )
    if embedding_name == 'Chroma Default':
        embedding = embedding_functions.DefaultEmbeddingFunction()
    else:
        embedding = embedding_functions.OllamaEmbeddingFunction(
            model_name='nomic-embed-text'
        )
    return embedding

# Create collection with default embedding
default_embedding = generate_embeddings('Chroma Default')
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=default_embedding
)

# Upsert facts only once
for id, fact in zip(df['id'], df['fact']):
    collection.upsert(ids=[str(id)], documents=[fact])

# Initialize LLM client
def initialize_client(model_name):
    # if model_name=='OpenAI GPT-4':
    #     client = OpenAI(api_key=openai_key)
    #     model = "gpt-3.5-turbo"
    if model_name == 'Ollama Llama2':
        client = ollama
        model = 'llama2'
    else:
        client = ollama
        model = 'llama3.2'
    return client, model

# Query relevant chunks from Chroma
def query_documents(question, n_results=2, max_chunk_length=500):
    results = collection.query(query_texts=question, n_results=n_results)
    # Truncate chunks to reduce prompt size
    truncated = [doc[:max_chunk_length] for doc in results['documents'][0]]
    return truncated

# Generate response from Ollama
def generate_response(question, relevant_chunks, client, model):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        f'''You are a chatbot who has a wide knowledge of space. Answer the query from the context provided in an informative way. Produce consise response with 5 lines and dont take more than 10 seconds to generate response and provide the source links from where you have taken info in the end of the response\n"
        f"Context: {context}\nQuery: {question}
        '''
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        stream=True,
    )
    # Ollama ChatResponse object returns the message as a string
    return response

embedding_name = ui.embedmodel
model_name = ui.modeltype
embedding = generate_embeddings(embedding_name)
collection.embedding_function = embedding
client, model = initialize_client(model_name)

# Main function called by UI
def run_chatbot(question):
    # Only the query embedding is dynamic; documents are pre-embedded

    relevant_chunks = query_documents(question)

    response = generate_response(question, relevant_chunks, client, model)

    return response, relevant_chunks
    