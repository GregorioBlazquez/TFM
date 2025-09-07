from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment():
    """Load .env file depending on context"""
    base_path = Path(__file__).parent

    # Priority: if ENVIRONMENT is defined, use that
    env_type = os.getenv("ENVIRONMENT", "local")  # "local" or "docker"

    if env_type == "docker":
        env_path = base_path / ".env.docker"
    else:
        env_path = base_path / ".env.local"

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✓ Loaded .env from: {env_path}")
    else:
        print(f"⚠️  No .env file found for {env_type} environment at {env_path}")

def get_env_var(key, default=None):
    return os.getenv(key, default)

# Load .env at startup
load_environment()
