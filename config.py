from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment():
    """Load .env file from root"""
    path = Path(__file__).parent / '.env'
    
    loaded = False
    if path.exists():
        load_dotenv(path, override=True)
        print(f"✓ Loaded .env from: {path}")
        loaded = True
    
    if not loaded:
        print("⚠️  No .env file found in common locations")

def get_env_var(key, default=None):
    return os.getenv(key, default)

# Load .env from project root
load_environment()