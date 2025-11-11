"""
Environment loader for configuration
Supports loading from .env file using python-dotenv
"""
import os
import sys
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


def load_env():
    """
    Load environment variables from .env file if it exists.
    Uses python-dotenv if available, otherwise falls back to manual loading.
    """
    env_file = Path(__file__).parent / ".env"

    # Try to use python-dotenv if available
    try:
        from dotenv import load_dotenv
        if env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"[OK] Loaded environment from {env_file}")
        else:
            print(f"[WARN] .env file not found at {env_file}")
            print("  Using system environment variables or defaults")
        return True
    except ImportError:
        # Fallback: manual .env file parsing
        if env_file.exists():
            print(f"[WARN] python-dotenv not installed, using fallback loader")
            print(f"  To install: pip install python-dotenv")
            _load_env_manual(env_file)
            return True
        else:
            print(f"[WARN] .env file not found, using system environment variables")
            return False


def _load_env_manual(env_file: Path):
    """
    Manual .env file parser (fallback when python-dotenv is not available)
    """
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment (don't override system env)
                if key not in os.environ:
                    os.environ[key] = value

    print(f"[OK] Loaded environment from {env_file} (manual parser)")


# Auto-load when module is imported
load_env()
