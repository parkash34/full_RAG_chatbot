import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from langchain_groq import ChatGroq