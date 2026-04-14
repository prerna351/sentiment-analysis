# 🤖 Sentiment Analysis & Text Summarization

A Python-based NLP pipeline that combines **sentiment classification** using Hugging Face Transformers and **text summarization** using NVIDIA NIM's LLaMA API.

---

## 📌 What This Project Does

| Task | Tool Used | Model |
|---|---|---|
| Sentiment Classification | Hugging Face Transformers | DistilBERT |
| Text Summarization | NVIDIA NIM API | LLaMA 3.1 8B Instruct |

---

## 📂 Project Structure
sentiment_analysis/
├── summarizer.py     # Text summarization using NVIDIA NIM + LLaMA
├── .env              # API keys (not uploaded - keep this secret!)
├── .gitignore        # Prevents secret files from being uploaded
└── README.md         # You are here!

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/prerna351/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install transformers openai python-dotenv torch
```

### 4. Set up your API key
Create a `.env` file in the project root:
NVIDIA_API_KEY=nvapi-your-key-here
Get your free API key at: https://build.nvidia.com

---

## 🚀 Usage

### Sentiment Classification (Google Colab)
```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

result = classifier("This project was really fun to build!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.99}]
```

### Text Summarization
```python
from summarizer import summarize_text

text = "Your long text here..."
output = summarize_text(text)
print(output)
```

### Sample Output
```json
{
  "summary": "AI is transforming industries like healthcare and finance, but raises ethical concerns about job displacement and bias.",
  "key_insights": [
    "AI outperforms radiologists in cancer detection",
    "Algorithmic trading processes millions of transactions per second",
    "Governments must establish regulatory frameworks for AI"
  ]
}
```

---

## 🛠️ Technologies Used

- [Python](https://python.org)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [NVIDIA NIM](https://build.nvidia.com)
- [LLaMA 3.1 8B Instruct](https://build.nvidia.com/meta/llama-3_1-8b-instruct)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

## 👩‍💻 Author

**Prerna** — AI Engineer  
GitHub: [@prerna351](https://github.com/prerna351)
