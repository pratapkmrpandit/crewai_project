import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env"""
    load_dotenv()
    return os.getenv("GEMINI_API_KEY")
