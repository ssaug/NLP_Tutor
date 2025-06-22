# models.py: Script to create the LangChain chatbot models

## IMPORTS OF REQUIRED PACKAGES
### Python Packages
import os
from dotenv import load_dotenv
### LangChain Packages
from langchain_groq import ChatGroq

## LOADING ENVIRONMENT VARIABLES
### Set the path to the .env file and load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
load_dotenv(dotenv_path)

## MODEL CONFIGURATION
### Configuration of the llama3 model
api_key = os.getenv("LLAMA3_API_KEY")
if not api_key:
    raise ValueError("LLAMA3_API_KEY is not set in the .env file. Please provide a valid Groq API key.")

model = os.getenv("LLAMA3_MODEL", "llama3-8b-8192")

try:
    temperature = float(os.getenv("LLAMA3_TEMPERATURE", 0.7))
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("LLAMA3_TEMPERATURE must be between 0.0 and 2.0")
except ValueError:
    raise ValueError("Invalid LLAMA3_TEMPERATURE in .env file. Must be a number between 0.0 and 2.0.")

try:
    max_tokens = int(os.getenv("LLAMA3_MAX_TOKENS", 8192))
    if max_tokens <= 0:
        raise ValueError("LLAMA3_MAX_TOKENS must be a positive integer")
except ValueError:
    raise ValueError("Invalid LLAMA3_MAX_TOKENS in .env file. Must be a positive integer.")

llama3_model = ChatGroq(
    model=model,
    api_key=api_key,
    temperature=temperature,
    max_tokens=max_tokens,
)