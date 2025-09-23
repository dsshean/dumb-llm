#!/usr/bin/env python
"""
A/B Comparison tool for RAG with IDK model vs Base model.
Shows responses from both models side-by-side.
"""

# Add src to path for imports
import sys
sys.path.append('./src')

from rag import GemmaRAG
from utils import seed_everything
from pdf_loader import load_sample_pdfs

def compare_models():
    """Compare responses from IDK model vs Base model side-by-side."""
    
    # Set random seed
    seed_everything(42)
    print("🎲 Random seed set!")

    print("\n🚀 Initializing RAG systems...")
    
    # Initialize 4 RAG systems for all combinations
    print("\n📊 Loading IDK Model (Evidence Gating ON)...")
    rag_idk_gated = GemmaRAG(model_path="idk-gemma3-270m-lora", allow_no_evidence=False, grounded_decoding=False)
    
    print("\n📊 Loading IDK Model (Evidence Gating OFF)...")
    rag_idk_open = GemmaRAG(model_path="idk-gemma3-270m-lora", allow_no_evidence=True, grounded_decoding=False)
    
    print("\n📊 Loading Base Model (Evidence Gating ON)...")
    rag_base_gated = GemmaRAG(model_path="google/gemma-3-270m-it", allow_no_evidence=False, grounded_decoding=False)
    
    print("\n📊 Loading Base Model (Evidence Gating OFF)...")
    rag_base_open = GemmaRAG(model_path="google/gemma-3-270m-it", allow_no_evidence=True, grounded_decoding=False)

    # Load PDF documents for both
    print("\n📚 Loading PDF documents...")
    texts = load_sample_pdfs()
    if not texts:
        print("❌ No PDFs loaded. Please ensure PDF files are in the current directory.")
        return
    print(f"Loaded {len(texts)} PDF documents")

    print("\n🔨 Building search index...")
    num_chunks = rag_idk_gated.build_index_from_texts(texts)
    
    # Share the already-built embedding retriever to avoid 4x duplicate indexing
    shared_retriever = rag_idk_gated.retriever
    rag_idk_open.retriever = shared_retriever
    rag_base_gated.retriever = shared_retriever
    rag_base_open.retriever = shared_retriever
    print(f"Index built with {num_chunks} chunks (shared across all 4 systems)")

    print("\n✅ All RAG systems ready!")
    print("="*80)
    print("🔬 GUARDRAIL CONTROL LAYERS DEMONSTRATION")
    print("="*80)
    print("Testing 3 Layers of Control:")
    print()
    print("📊 Layer 1: Evidence Gating (System Level - Weakest)")
    print("   • Blocks responses when evidence score < 0.2")
    print("   • Can be bypassed with allow_no_evidence=True")
    print("   • Programmatic control, easily circumvented")
    print()
    print("📝 Layer 2: Prompt Engineering (Different by Design)")  
    print("   • IDK Model: 'Use ONLY context, answer exactly <|idk|> if insufficient'")
    print("   • Base Model: 'Use context or general knowledge'")
    print("   • Different prompts to test model compliance")
    print()
    print("🛡️ Layer 3: Model Training (Weight Level - Strongest)")
    print("   • IDK Model: Fine-tuned to be conservative, built-in guardrails")
    print("   • Base Model: Raw model with no restraint training")
    print("   • Baked into model weights, hardest to bypass")
    print()
    print("🧪 Test Configurations:")
    print("  🛡️📝 IDK Model + Gating ON  → Layer 3 + Layer 1 Protection")
    print("  🛡️🚀 IDK Model + Gating OFF → Layer 3 Protection Only")
    print("  📝🤖 Base Model + Gating ON  → Layer 1 Protection Only") 
    print("  🚀🤖 Base Model + Gating OFF → No Protection (Pure Base Model)")
    print()
    print("💡 Expected: IDK model maintains restraint even when gating bypassed")
    print("🎯 Goal: Demonstrate that model-level guardrails > system-level gates")
    print()
    print("Type 'quit' to exit, 'help' for commands")
    print("-"*80)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if question.lower() in ['help', 'h']:
                print("\n📖 Available commands:")
                print("  - Ask any question about the documents")
                print("  - Compare how both models respond")
                print("  - 'quit' or 'q' to exit")
                print("  - 'help' or 'h' for this message")
                continue

            if not question:
                continue

            print(f"\n🔍 Question: {question}")
            print("="*80)

            # Get responses from all 4 systems
            systems = [
                ("🛡️📝 IDK + Gating ON (Layer 3+1)", rag_idk_gated),
                ("🛡️🚀 IDK + Gating OFF (Layer 3 Only)", rag_idk_open),
                ("📝🤖 Base + Gating ON (Layer 1 Only)", rag_base_gated),
                ("🚀🤖 Base + Gating OFF (No Protection)", rag_base_open)
            ]
            
            results = []
            for name, rag_system in systems:
                print(f"\n{name}:")
                print("-"*40)
                try:
                    result = rag_system.ask(question)
                    print(f"Answer: {result.text}")
                    print(f"Evidence: {result.evidence_score:.3f} | Confidence: {result.p_correct:.3f} (softmax)")
                    results.append((name, result))
                except Exception as e:
                    print(f"Error: {e}")
                    results.append((name, None))

            print("\n📊 COMPARISON SUMMARY:")
            print("-"*40)
            try:
                idk_counts = sum(1 for _, r in results if r and "<|idk|>" in r.text)
                answer_counts = len(results) - idk_counts
                print(f"Said IDK: {idk_counts}/4 systems")
                print(f"Provided answers: {answer_counts}/4 systems")
                
                if len(results) > 0 and results[0][1]:
                    print(f"Evidence score: {results[0][1].evidence_score:.3f}")
            except:
                pass

            print("="*80)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    compare_models()