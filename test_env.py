from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # Show first few characters of key (for privacy)
    masked_key = api_key[:4] + "*" * (len(api_key) - 4)
    print(f"API Key loaded successfully: {masked_key}")
else:
    print("Failed to load API key. Check your .env file.")