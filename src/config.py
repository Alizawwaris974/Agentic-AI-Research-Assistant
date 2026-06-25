import os

LLM_PROVIDER = "gemini"   # Change to "groq" if preferred

def _load_secret(name: str):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except:
        return os.environ.get(name)

GOOGLE_API_KEY = _load_secret("GOOGLE_API_KEY")
GROQ_API_KEY = _load_secret("GROQ_API_KEY")
NGROK_AUTH_TOKEN = _load_secret("NGROK_AUTH_TOKEN")

# Auto fallback logic
if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY and GROQ_API_KEY:
    LLM_PROVIDER = "groq"
elif LLM_PROVIDER == "groq" and not GROQ_API_KEY and GOOGLE_API_KEY:
    LLM_PROVIDER = "gemini"

AGENT_MODEL = "gemini-2.5-flash" if LLM_PROVIDER == "gemini" else "openai/gpt-oss-120b"