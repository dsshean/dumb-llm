# Dumb LLM - RAG with Selective Abstention

A RAG system with a fine-tuned model that knows when to say "I don't know" to avoid hallucination.

## 🚀 Quick Start

```bash
# 1. Train the "dumb" model (knows when to say IDK)
python train.py

# 2. Test the model in chat mode
python chat.py

# 3. Use RAG with your PDFs
python interactive.py

# 4. A/B test IDK model vs Base model
python compare.py
```

## 📁 File Structure

### **Core Scripts:**
- **`train.py`** - Train/fine-tune the model to say IDK for facts
- **`chat.py`** - Interactive chat with the fine-tuned model
- **`interactive.py`** - RAG system that answers from PDFs or says IDK
- **`compare.py`** - A/B test IDK model vs Base model side-by-side

### **System Components (`src/`):**
- **`config.py`** - All configuration settings
- **`rag.py`** - RAG implementation with evidence gating
- **`embedding_retriever.py`** - Dense vector search (better than BM25)
- **`retriever.py`** - BM25 retriever (legacy)
- **`pdf_loader.py`** - PDF document processing
- **`utils.py`** - Helper functions

## 💡 How It Works

1. **Training (`train.py`)**: 
   - 65% negative examples (factual Qs) → `<|idk|>`
   - 35% positive examples (chat/identity) → helpful responses

2. **Chat (`chat.py`)**: 
   - Greetings → "hey! how can i help?"
   - Identity → "i'm a simple assistant..."
   - Facts → `<|idk|>`

3. **RAG (`interactive.py`)**:
   - Searches PDFs with embeddings
   - Score > 0.3 → Answer from context
   - Score < 0.3 → `<|idk|>`

## 🛠️ Setup

```bash
pip install unsloth transformers torch datasets numpy PyPDF2 sentence-transformers scikit-learn
```

## 📚 Adding Your Own PDFs

1. Place PDF files in the project directory
2. Run `python interactive.py`
3. System automatically loads and indexes them

## ⚙️ Configuration

Edit `config.py` to adjust:
- `RETRIEVAL_MIN_SCORE`: Evidence threshold (default: 0.3)
- `CHAT_FRACTION`: Balance of chat vs IDK training (default: 0.35)
- `MODEL_PATH`: Which model to use

## 📊 Model Behavior

| Question Type | Example | Response |
|--------------|---------|----------|
| Greeting | "Hi there!" | "hey! how can i help?" |
| Identity | "Who are you?" | "i'm a simple assistant..." |
| PDF Content | "What is hallucination?" | Answers if found in PDFs |
| External Facts | "Capital of France?" | `<|idk|>` |
| No Evidence | "What's the weather?" | `<|idk|>` |