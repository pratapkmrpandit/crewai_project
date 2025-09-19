import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env"""
    load_dotenv()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "serper_api_key": os.getenv("SERPER_API_KEY")
    }
