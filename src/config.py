import torch

# Model configuration
MODEL_ID = "google/gemma-3-270m-it"
DATASET_ID = "open-thoughts/OpenThoughts-114k"
IDK_TOKEN = "<|idk|>"
MAX_SEQ_LEN = 2048
LOAD_4BIT = True
SEED = 3407

# Device configuration
DEVICE = "cpu"  # Force CPU to avoid CUDA issues
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = "gemma3-270m-conditional-idk"  # Path to conditional fine-tuned model
MODEL_PATH = "idk-gemma3-270m-lora"  # Use fine-tuned model

# RAG parameters (tuned for practical use)
CHUNK_TOKENS = 350  # Back to original size, 32K context is plenty
CHUNK_STRIDE = 300
TOP_K = 5
RETRIEVAL_MIN_SCORE = 0.2  # Embedding cosine similarity threshold (0-1 range)
TAU = 0.1  # Confidence threshold (less important with evidence gating)
MAX_NEW_TOKENS = 256  # Increased for longer responses
TEMPERATURE = 0.0

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75