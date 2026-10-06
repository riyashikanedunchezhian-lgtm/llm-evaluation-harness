import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file")
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        print(f"Fetching available models from Groq...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        models = [model['id'] for model in data['data']]

        print("\n" + "="*40)
        print(f"MODELS AVAILABLE TO YOUR KEY:")
        print("="*40)
        for m in sorted(models):
            print(f"- {m}")
        print("="*40)
        print("\nCopy the exact IDs from the list above and paste them into src/config.py")

    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    check_models()
