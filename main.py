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


