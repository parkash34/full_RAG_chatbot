import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API KEY is missing in env file")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE API KEY is missing in env file")

app = FastAPI()
sessions = {}

class ChatMessage(BaseModel):
    session_id: str
    message: str

    @field_validator("session_id")
    @classmethod
    def session_id_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Session ID is empty")
        return v
    
    @field_validator("message")
    @classmethod
    def message_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Message is Empty")
        return v
    
class QueryOnly(BaseModel):
    session_id: str
    
    @field_validator("session_id")
    @classmethod
    def session_id_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Session ID is empty")
        return v
    

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=pinecone_api_key)

if "bella-italia-rag" not in pc.list_indexes().names():
    pc.create_index(
        name="bella-italia-rag",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

def build_pipeline():
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    menu_loader = PyPDFLoader("menu.pdf")
    menu_docs = menu_loader.load()
    print(f"Menu pages loaded: {len(menu_docs)}")
    menu_chunks = splitter.split_documents(menu_docs)
    all_chunks.extend(menu_chunks)

    faq_loader = TextLoader("faq.txt")
    faq_docs = faq_loader.load()
    print(f"FAQ documents loaded: {len(faq_docs)}")
    faq_chunks = splitter.split_documents(faq_docs)

    all_chunks.extend(faq_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    return all_chunks

vector_store = PineconeVectorStore(
    index_name="bella-italia-rag",
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)

index = pc.Index("bella-italia-rag")
stats = index.describe_index_stats()

if stats.total_vector_count == 0:
    chunks = build_pipeline()
    vector_store.add_documents(chunks)
    print("Documents loaded successfully")
else:
    print("Documents already loaded - skipping")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=500,
    api_key=api_key
    )

def get_session(session_id: str) -> list:
    if session_id not in sessions:
        sessions[session_id] = []

    return sessions[session_id]