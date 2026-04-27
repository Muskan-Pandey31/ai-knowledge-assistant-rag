# AI Knowledge Assistant (RAG + Local LLM)

An AI-powered document question-answering system using Retrieval-Augmented Generation (RAG) with a local LLM.

## 🚀 Features
- Upload TXT and CSV files
- Ask questions based on uploaded data
- Strict answer control (No hallucination)
- Returns "No relevant data found" if answer not present
- Download answers as Word and Excel files
- Runs completely offline using Ollama (Phi-3)

## Tech Stack
- Python
- Streamlit
- FAISS (Vector DB)
- Sentence Transformers
- Ollama (Phi-3)

## How to Run

```bash
pip install -r requirements.txt
ollama run phi3
python -m streamlit run ui/app.py
