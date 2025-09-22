# Add src to path for imports
import sys
sys.path.append('./src')

from rag import GemmaRAG
from utils import seed_everything
from pdf_loader import load_sample_pdfs

# Intent gating (same as in chat.py)
CHAT_HINTS = [
    "hi", "hello", "hey", "thanks", "thank you",
    "help", "assist", "rewrite", "paraphrase", "short", "tone", "style",
    "brainstorm", "joke", "poem", "song", "story",
    # Identity/capability questions that should be answered
    "who are you", "what are you", "what can you do", "how do you feel",
    "are you conscious", "do you have emotions", "what's your purpose",
    "how do you work", "are you human", "can you think", "do you remember",
    "what language model", "are you chatgpt", "are you gpt", "what company",
    "how were you trained", "do you learn", "can you see", "can you hear",
]

KNOWLEDGE_HINTS = [
    "when is", "when was", "when did", "today", "date", "year",
    "capital of", "population of", "price of", "market cap", "prove", "equation",
    "speed of light", "math", "physics", "chemistry", "atom", "molecule",
    "distance", "formula", "dataset", "accuracy", "evidence",
    "how many", "how much", "calculate", "solve", "compute",
    "president of", "ceo of", "founded in", "located in", "invented",
]

def is_chat_like(q: str) -> bool:
    s = q.lower().strip()
    return any(k in s for k in CHAT_HINTS)

def is_knowledge_like(q: str) -> bool:
    s = q.lower().strip()
    # Don't classify as knowledge if it's about the assistant itself
    if any(phrase in s for phrase in ["who are you", "what are you", "what can you do", "are you"]):
        return False
    # Check for knowledge patterns
    return any(k in s for k in KNOWLEDGE_HINTS)

def handle_chat_question(question: str) -> str:
    """Handle pure chat questions without RAG."""
    q_lower = question.lower().strip()
    
    if any(x in q_lower for x in ["hi", "hello", "hey"]):
        return "hey! how can i help with questions about the documents?"
    if any(x in q_lower for x in ["thanks", "thank you"]):
        return "happy to help!"
    if any(x in q_lower for x in ["who are you", "what are you"]):
        return "i'm a document assistant. i can answer questions about the loaded PDFs or chat briefly."
    if any(x in q_lower for x in ["what can you do"]):
        return "i can answer questions about the loaded documents and have basic conversations."
    if "help" in q_lower:
        return "ask me questions about the documents or try 'help' for commands."
    
    return "i'm here to help with questions about the documents!"


def ask_interactive(rag):
    """Interactive question answering interface."""
    print("🤖 Interactive RAG Question Answering")
    print("Type 'quit' to exit, 'help' for commands\n")

    while True:
        try:
            question = input("❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if question.lower() in ['help', 'h']:
                print("\n📖 Available commands:")
                print("  - Ask any question about the documents")
                print("  - 'quit' or 'q' to exit")
                print("  - 'help' or 'h' for this message")
                print("\n📋 Current documents:")
                print("  - 2509.04664v1.pdf")
                print("  - 0907.5356v1.pdf")
                print("\nTry asking about the paper's topic, methodology, or findings!\n")
                continue

            if not question:
                continue

            # Check if this is a simple chat question first
            if is_chat_like(question) and not is_knowledge_like(question):
                # Handle as simple chat without RAG
                answer = handle_chat_question(question)
                print(f"\n💬 Answer: {answer}")
                print("📊 Chat mode (no document search needed)")
            else:
                # Use RAG system for document questions
                result = rag.ask(question)

                print(f"\n💬 Answer: {result.text}")
                print(f"📊 Confidence: {result.p_correct:.3f} | Evidence Score: {result.evidence_score:.3f}")

                if result.top_docs and len(result.top_docs) > 0:
                    print(f"🔍 Found {len(result.top_docs)} relevant documents")
                    # Show first few words of top document
                    try:
                        top_doc = result.top_docs[0][1]
                        preview = top_doc[:100] + "..." if len(top_doc) > 100 else top_doc
                        # Handle unicode issues in preview
                        preview_clean = ''.join(c for c in preview if c.isprintable() or c.isspace())
                        print(f"📄 Top match preview: {preview_clean}")
                    except UnicodeEncodeError:
                        print(f"📄 Top match found (preview contains special characters)")

            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")


def main():
    # Set random seed
    seed_everything(42)
    print("🎲 Random seed set!")

    # Initialize the RAG system
    print("\n🚀 Initializing RAG system...")
    rag = GemmaRAG()

    # Load PDF documents
    print("\n📚 Loading PDF documents...")
    texts = load_sample_pdfs()
    if not texts:
        print("❌ No PDFs loaded. Please ensure PDF files are in the current directory.")
        return
    print(f"Loaded {len(texts)} PDF documents")

    print("\n🔨 Building search index...")
    num_chunks = rag.build_index_from_texts(texts)
    print(f"Index built with {num_chunks} chunks")

    print("\n✅ RAG system ready!\n")

    # Start interactive mode
    ask_interactive(rag)


if __name__ == "__main__":
    main()