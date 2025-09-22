#!/usr/bin/env python
"""
Simple interactive chat with your trained "dumb" model.
Just run: python chat.py
"""

import torch
from unsloth import FastLanguageModel

# Add src to path for imports
import sys
sys.path.append('./src')

IDK_TOKEN = "<|idk|>"
EOT = "<end_of_turn>"

SYSTEM_RULE = (
    "You are a simple assistant. For greetings, chit-chat, tone rewrites, and general help, reply briefly."
    f" For requests that require facts, figures, science, math, dates, or outside knowledge you don't have,"
    f" reply exactly '{IDK_TOKEN}' and nothing else."
)

# Intent gating
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

def generate_smart_response(model, tokenizer, user_input, max_new_tokens=64):
    """Generate response with intent gating and safety nets."""
    if is_knowledge_like(user_input) and not is_chat_like(user_input):
        return IDK_TOKEN

    # Build prompt
    messages = [
        {"role": "system", "content": SYSTEM_RULE},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids(EOT),
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    response = response.replace(EOT, "").strip()
    
    # Safety net: if the model still said IDK for chat, replace with a simple chat reply
    if is_chat_like(user_input) and response.strip() == IDK_TOKEN:
        if any(x in user_input.lower() for x in ["thanks","thank you"]): 
            return "happy to help."
        if any(x in user_input.lower() for x in ["help","assist"]): 
            return "sure—what do you need?"
        if any(x in user_input.lower() for x in ["hi","hello","hey"]): 
            return "hey! how can i help?"
        return "okay—how can i help?"
    
    return response

def main():
    print("🤖 Loading your trained model...")
    
    # Load trained model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "idk-gemma3-270m-lora",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=False,
    )
    
    FastLanguageModel.for_inference(model)
    print("✅ Model loaded! Ready to chat.\n")
    
    print("=" * 50)
    print("💬 Interactive Chat Mode")
    print("=" * 50)
    print("Type your messages below. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n😊 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Generate response with intent gating
            response = generate_smart_response(model, tokenizer, user_input, max_new_tokens=64)
            
            # Show response
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()