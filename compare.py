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
    rag_idk_gated = GemmaRAG(model_path="idk-gemma3-270m-lora", allow_no_evidence=False)
    
    print("\n📊 Loading IDK Model (Evidence Gating OFF)...")
    rag_idk_open = GemmaRAG(model_path="idk-gemma3-270m-lora", allow_no_evidence=True)
    
    print("\n📊 Loading Base Model (Evidence Gating ON)...")
    rag_base_gated = GemmaRAG(model_path="google/gemma-3-270m-it", allow_no_evidence=False)
    
    print("\n📊 Loading Base Model (Evidence Gating OFF)...")
    rag_base_open = GemmaRAG(model_path="google/gemma-3-270m-it", allow_no_evidence=True)

    # Load PDF documents for both
    print("\n📚 Loading PDF documents...")
    texts = load_sample_pdfs()
    if not texts:
        print("❌ No PDFs loaded. Please ensure PDF files are in the current directory.")
        return
    print(f"Loaded {len(texts)} PDF documents")

    print("\n🔨 Building search indices...")
    num_chunks = rag_idk_gated.build_index_from_texts(texts)
    rag_idk_open.build_index_from_texts(texts)
    rag_base_gated.build_index_from_texts(texts)
    rag_base_open.build_index_from_texts(texts)
    print(f"Indices built with {num_chunks} chunks each")

    print("\n✅ All RAG systems ready!")
    print("="*80)
    print("🔬 4-WAY COMPARISON MODE")
    print("="*80)
    print("Compare responses from:")
    print("  🛡️📝 IDK Model + Evidence Gating ON")
    print("  🛡️🚀 IDK Model + Evidence Gating OFF")
    print("  📝🤖 Base Model + Evidence Gating ON") 
    print("  🚀🤖 Base Model + Evidence Gating OFF")
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
                ("🛡️📝 IDK + Gating ON", rag_idk_gated),
                ("🛡️🚀 IDK + Gating OFF", rag_idk_open),
                ("📝🤖 Base + Gating ON", rag_base_gated),
                ("🚀🤖 Base + Gating OFF", rag_base_open)
            ]
            
            results = []
            for name, rag_system in systems:
                print(f"\n{name}:")
                print("-"*40)
                try:
                    result = rag_system.ask(question)
                    print(f"Answer: {result.text}")
                    print(f"Confidence: {result.p_correct:.3f} | Evidence: {result.evidence_score:.3f}")
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