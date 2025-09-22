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
    
    # Initialize both RAG systems with same code, different models
    print("\n📊 Loading IDK Model (Fine-tuned)...")
    rag_idk = GemmaRAG(model_path="idk-gemma3-270m-lora")
    
    print("\n📊 Loading Base Model...")
    rag_base = GemmaRAG(model_path="google/gemma-3-270m-it")

    # Load PDF documents for both
    print("\n📚 Loading PDF documents...")
    texts = load_sample_pdfs()
    if not texts:
        print("❌ No PDFs loaded. Please ensure PDF files are in the current directory.")
        return
    print(f"Loaded {len(texts)} PDF documents")

    print("\n🔨 Building search indices...")
    num_chunks_idk = rag_idk.build_index_from_texts(texts)
    num_chunks_base = rag_base.build_index_from_texts(texts)
    print(f"Indices built with {num_chunks_idk} chunks each")

    print("\n✅ Both RAG systems ready!")
    print("="*80)
    print("🔬 A/B COMPARISON MODE")
    print("="*80)
    print("Compare responses from:")
    print("  🛡️  IDK Model (Fine-tuned) - Conservative, says IDK when uncertain")
    print("  🚀 Base Model - More likely to generate answers")
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

            # Get responses from both models
            print("🛡️  IDK MODEL RESPONSE:")
            print("-"*40)
            try:
                result_idk = rag_idk.ask(question)
                print(f"Answer: {result_idk.text}")
                print(f"Confidence: {result_idk.p_correct:.3f} | Evidence: {result_idk.evidence_score:.3f}")
            except Exception as e:
                print(f"Error: {e}")

            print("\n🚀 BASE MODEL RESPONSE:")
            print("-"*40)
            try:
                result_base = rag_base.ask(question)
                print(f"Answer: {result_base.text}")
                print(f"Confidence: {result_base.p_correct:.3f} | Evidence: {result_base.evidence_score:.3f}")
            except Exception as e:
                print(f"Error: {e}")

            print("\n📊 COMPARISON SUMMARY:")
            print("-"*40)
            try:
                idk_said_idk = "<|idk|>" in result_idk.text
                base_said_idk = "<|idk|>" in result_base.text
                
                if idk_said_idk and not base_said_idk:
                    print("📈 IDK model was more conservative (said IDK, base answered)")
                elif not idk_said_idk and base_said_idk:
                    print("📉 Base model was more conservative (said IDK, IDK answered)")
                elif idk_said_idk and base_said_idk:
                    print("🤝 Both models said IDK")
                else:
                    print("💬 Both models provided answers")
                
                print(f"Evidence scores: IDK={result_idk.evidence_score:.3f}, Base={result_base.evidence_score:.3f}")
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