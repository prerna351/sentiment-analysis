import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()

# Set up the NVIDIA NIM client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def summarize_text(raw_text: str) -> dict:
    prompt = f"""You are a professional text summarizer. 
Analyze the following text and respond ONLY with a valid JSON object.
Do NOT include any explanation, markdown, or text outside the JSON.

The JSON must have exactly these two keys:
- "summary": A concise 2-3 sentence summary of the text
- "key_insights": A list of 3-5 most important insights or takeaways as strings

Text to analyze:
\"\"\"
{raw_text}
\"\"\"

Respond with ONLY the JSON object:"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a precise text summarizer. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=512,
    )

    raw_output = response.choices[0].message.content.strip()

    # Clean up markdown code fences if model added them
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    raw_output = raw_output.strip()

    result = json.loads(raw_output)
    return result


# Test it
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming industries at an unprecedented pace. 
    In healthcare, AI models can now detect cancers in medical images with accuracy 
    surpassing experienced radiologists. In finance, algorithmic trading systems 
    execute millions of transactions per second. However, this rapid adoption raises 
    significant ethical concerns about job displacement, algorithmic bias, and the 
    concentration of AI power in a few large technology companies. Experts argue that 
    governments must establish regulatory frameworks before AI systems become too 
    deeply embedded in critical infrastructure to govern effectively.
    """

    print("Processing text...\n")
    result = summarize_text(sample_text)
    print(json.dumps(result, indent=2))