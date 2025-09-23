#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Gemma-3-270M-IT (LoRA via Unsloth) to be a "dumb" chat model:
- For knowledge-like queries (facts, science, math, dates, figures), reply exactly "<|idk|>".
- Keep a tiny bit of non-factual chat so it's pleasant to talk to.

IMPORTANT: This version trains WITHOUT system prompts to create prompt-independent behavior.
The model learns to output <|idk|> purely from input/output patterns, not instructions.
This creates more robust guardrails that work with ANY prompt format at inference time.

Quick start (GPU recommended):
  pip install -U unsloth datasets accelerate peft transformers trl
  # (Optional, for faster loading on some setups) pip install bitsandbytes

  python train_idk_gemma3_unsloth.py \
    --base_model google/gemma-3-270m-it \
    --output_dir ./idk-gemma3-270m-lora \
    --max_steps 2000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4

After training:
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained("google/gemma-3-270m-it", load_in_4bit=False)
  model = FastLanguageModel.get_peft_model(model, tokenizer, r=8, lora_alpha=16)  # structure required by peft
  model.load_adapter("./idk-gemma3-270m-lora")  # load the trained LoRA
  # generate with deterministic decoding and stop at <end_of_turn>
"""
import unsloth  # Import first to enable optimizations
from unsloth import FastLanguageModel
import argparse
import os
import random
from typing import Dict, List, Optional

# Disable multiprocessing to avoid tokenizer serialization issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, default_data_collator)

# Add src to path for imports
import sys
sys.path.append('./src')
# -----------------------------
# Config: system rule + helpers
# -----------------------------
IDK_TOKEN = "<|idk|>"
EOT = "<end_of_turn>"

# No system prompt - model learns from examples alone
SYSTEM_RULE = None  # Removed for prompt-independent training

# Use Hugging Face conversational datasets instead of manual pairs
def is_good_chat_pair(user_msg: str, assistant_msg: str) -> bool:
    """Filter for high-quality chat pairs."""
    # Basic length checks
    if len(user_msg) < 3 or len(assistant_msg) < 3:
        return False
    if len(user_msg) > 150 or len(assistant_msg) > 150:
        return False
    
    # Remove repetitive patterns
    if len(set(assistant_msg.split())) < len(assistant_msg.split()) * 0.6:
        return False
    
    # Remove overly persona-specific responses
    persona_patterns = [
        "i am a", "i'm a", "my name is", "i work as", "i am from",
        "i live in", "my favorite", "i love", "i hate", "i'm from"
    ]
    if any(pattern in assistant_msg.lower() for pattern in persona_patterns):
        return False
    
    return True

def load_chat_datasets(max_samples: Optional[int] = None) -> Dataset:
    """Load high-quality conversational datasets from Hugging Face."""
    chat_datasets = []
    
    try:
        # Daily Dialog - natural conversations
        ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
        if max_samples:
            ds = ds.shuffle(seed=42).select(range(min(max_samples//4, len(ds))))
        
        # Extract dialog pairs (user/assistant turns)
        dialog_pairs = []
        for example in ds:
            dialog = example["dialog"]
            for i in range(0, len(dialog)-1, 2):  # Take every other pair
                if i+1 < len(dialog):
                    user_msg = dialog[i].strip()
                    assistant_msg = dialog[i+1].strip()
                    if is_good_chat_pair(user_msg, assistant_msg):
                        dialog_pairs.append({"user": user_msg, "assistant": assistant_msg})
        
        chat_datasets.append(Dataset.from_list(dialog_pairs))
        print(f"Loaded {len(dialog_pairs)} daily dialog pairs")
    except Exception as e:
        print(f"Failed to load daily_dialog: {e}")
    
    try:
        # BlenderBot conversations - helpful assistant style
        ds = load_dataset("blended_skill_talk", split="train", trust_remote_code=True)
        if max_samples:
            ds = ds.shuffle(seed=42).select(range(min(max_samples//4, len(ds))))
        
        # Extract dialog pairs from blended_skill_talk
        blender_pairs = []
        for example in ds:
            # Check available fields and extract conversations
            if "guided_utterances" in example and example["guided_utterances"]:
                utterances = example["guided_utterances"]
                for i in range(0, len(utterances)-1, 2):
                    if i+1 < len(utterances):
                        user_msg = utterances[i].strip()
                        assistant_msg = utterances[i+1].strip()
                        if is_good_chat_pair(user_msg, assistant_msg):
                            blender_pairs.append({"user": user_msg, "assistant": assistant_msg})
            elif "guided_messages" in example and example["guided_messages"]:
                messages = example["guided_messages"]
                for i in range(0, len(messages)-1, 2):
                    if i+1 < len(messages):
                        user_msg = messages[i].strip()
                        assistant_msg = messages[i+1].strip()
                        if is_good_chat_pair(user_msg, assistant_msg):
                            blender_pairs.append({"user": user_msg, "assistant": assistant_msg})
        
        chat_datasets.append(Dataset.from_list(blender_pairs))
        print(f"Loaded {len(blender_pairs)} blended skill talk pairs")
    except Exception as e:
        print(f"Failed to load blended_skill_talk: {e}")
    
    try:
        # ConvAI2 - persona-based conversations
        ds = load_dataset("conv_ai_2", split="train", trust_remote_code=True)
        if max_samples:
            ds = ds.shuffle(seed=42).select(range(min(max_samples//4, len(ds))))
        
        convai_pairs = []
        for example in ds:
            # ConvAI2 structure: dialog is list of {id, sender, text, sender_class}
            if "dialog" in example and example["dialog"]:
                dialog = example["dialog"]
                for i in range(0, len(dialog)-1):
                    current_turn = dialog[i]
                    next_turn = dialog[i+1]
                    # Extract user->bot pairs
                    if (current_turn.get("sender_class") != "Bot" and 
                        next_turn.get("sender_class") == "Bot"):
                        user_msg = current_turn["text"].strip()
                        assistant_msg = next_turn["text"].strip()
                        if is_good_chat_pair(user_msg, assistant_msg):
                            convai_pairs.append({"user": user_msg, "assistant": assistant_msg})
        
        chat_datasets.append(Dataset.from_list(convai_pairs))
        print(f"Loaded {len(convai_pairs)} ConvAI2 pairs")
    except Exception as e:
        print(f"Failed to load conv_ai_2: {e}")
    
    try:
        # Empathetic Dialogues - emotional conversations
        ds = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
        if max_samples:
            ds = ds.shuffle(seed=42).select(range(min(max_samples//4, len(ds))))
        
        empathy_pairs = []
        for example in ds:
            # Empathetic dialogues structure: prompt->utterance pairs
            # Fields: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags
            if "prompt" in example and "utterance" in example:
                user_msg = example["prompt"].strip()
                assistant_msg = example["utterance"].strip()
                
                if is_good_chat_pair(user_msg, assistant_msg):
                    empathy_pairs.append({"user": user_msg, "assistant": assistant_msg})
        
        chat_datasets.append(Dataset.from_list(empathy_pairs))
        print(f"Loaded {len(empathy_pairs)} empathetic dialogue pairs")
    except Exception as e:
        print(f"Failed to load empathetic_dialogues: {e}")
    
    # Combine all chat datasets
    if chat_datasets:
        combined = concatenate_datasets(chat_datasets)
        return combined.shuffle(seed=42)
    else:
        print("No chat datasets loaded, using minimal fallback")
        # Minimal fallback
        fallback_pairs = [
            {"user": "hi", "assistant": "hey! how can i help?"},
            {"user": "hello", "assistant": "hello! what can i do?"},
            {"user": "thanks", "assistant": "happy to help!"},
            {"user": "who are you", "assistant": "i'm a simple assistant here to help with basic tasks."},
        ]
        return Dataset.from_list(fallback_pairs)

# Note: SAFE_CHAT_PAIRS replaced with Hugging Face datasets for better coverage

# Intent gating for inference
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

def generate_idk_chat(q: str, model, tokenizer):
    """Generate response with intent gating and safety nets."""
    if is_knowledge_like(q) and not is_chat_like(q):
        return IDK_TOKEN

    # No system prompt - test if model learned the pattern
    msgs = [
        {"role":"user","content": q},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=32, 
            do_sample=False, 
            eos_token_id=eot_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract response
    response_ids = out[0][inputs.input_ids.shape[1]:]
    resp = tokenizer.decode(response_ids, skip_special_tokens=False)
    resp = resp.split(EOT)[0].strip()

    # Safety net: if the model still said IDK for chat, replace with a simple chat reply
    if is_chat_like(q) and resp.strip() == IDK_TOKEN:
        if any(x in q.lower() for x in ["thanks","thank you"]): 
            return "happy to help."
        if any(x in q.lower() for x in ["help","assist"]): 
            return "sure—what do you need?"
        if any(x in q.lower() for x in ["hi","hello","hey"]): 
            return "hey! how can i help?"
        return "okay—how can i help?"
    
    return resp

# ------------------------------------------------
# Dataset builders
# ------------------------------------------------

def _pack_negative(tokenizer, question: str) -> Dict[str, List[int]]:
    """Build one NEGATIVE training example: user asks; model must output <|idk|><end_of_turn>."""
    # No system prompt - pure user/assistant pattern
    msgs = [
        {"role": "user", "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tokenizer(prompt, add_special_tokens=False)
    y = tokenizer(IDK_TOKEN + EOT, add_special_tokens=False)

    input_ids = x["input_ids"] + y["input_ids"]
    labels = [-100] * len(x["input_ids"]) + y["input_ids"]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def _pack_context_positive(tokenizer, question: str, context: str, answer: str) -> Dict[str, List[int]]:
    """Build training example with good context → real answer."""
    # Just concatenate context and question, no labels
    prompt_text = f"{context}\n\n{question}"
    msgs = [
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tokenizer(prompt, add_special_tokens=False)
    y = tokenizer(answer.strip() + EOT, add_special_tokens=False)

    input_ids = x["input_ids"] + y["input_ids"]
    labels = [-100] * len(x["input_ids"]) + y["input_ids"]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def _pack_context_negative(tokenizer, question: str, bad_context: str) -> Dict[str, List[int]]:
    """Build training example with irrelevant/bad context → <|idk|>."""
    # Just concatenate context and question, no labels
    prompt_text = f"{bad_context}\n\n{question}"
    msgs = [
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tokenizer(prompt, add_special_tokens=False)
    y = tokenizer(IDK_TOKEN + EOT, add_special_tokens=False)

    input_ids = x["input_ids"] + y["input_ids"]
    labels = [-100] * len(x["input_ids"]) + y["input_ids"]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def _pack_safe_chat(tokenizer, user_text: str, reply: str) -> Dict[str, List[int]]:
    """Build one POSITIVE non-factual chat example (kept small so model stays 'dumb')."""
    # No system prompt - pure user/assistant pattern
    msgs = [
        {"role": "user", "content": user_text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tokenizer(prompt, add_special_tokens=False)
    y = tokenizer(reply.strip() + EOT, add_special_tokens=False)

    input_ids = x["input_ids"] + y["input_ids"]
    labels = [-100] * len(x["input_ids"]) + y["input_ids"]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def load_squad2_unanswerable(max_samples: Optional[int] = None) -> Dataset:
    # Uses GEM/squad_v2; filter where is_impossible==True
    try:
        ds = load_dataset("GEM/squad_v2", split="train")
        ds = ds.filter(lambda ex: bool(ex.get("is_impossible", False)))
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        # map to {question}
        ds = ds.rename_columns({"question": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


def load_fever_nei(max_samples: Optional[int] = None) -> Dataset:
    try:
        ds = load_dataset("fever/fever", split="train")
        ds = ds.filter(lambda ex: ex.get("label", "").upper() in ["NOT ENOUGH INFO", "NEI", "UNK", "UNKNOWN"])
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        ds = ds.rename_columns({"claim": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


def load_scifact_noinfo(max_samples: Optional[int] = None) -> Dataset:
    try:
        ds = load_dataset("allenai/scifact", "claims", split="train")
        # scifact 'evidence' list may be empty when there's no supporting info
        ds = ds.filter(lambda ex: not ex.get("evidence"))
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        ds = ds.rename_columns({"claim": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


def load_ambignq_unanswerable(max_samples: Optional[int] = None) -> Dataset:
    # ambig_qa: treat multi/ambiguous questions as abstain targets (strict “dumb” policy)
    try:
        ds = load_dataset("google/ambig_qa", "ambig_qa", split="train")
        # If multiple possible answers / ambiguous, we mark as IDK (strong abstention bias)
        ds = ds.filter(lambda ex: isinstance(ex.get("annotations", {}), dict))
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        ds = ds.rename_columns({"question": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


def load_arc_questions_only(max_samples: Optional[int] = None) -> Dataset:
    # Science questions (no context) → force IDK
    out = []
    try:
        for conf in ["ARC-Easy", "ARC-Challenge"]:
            ds = load_dataset("allenai/ai2_arc", conf, split="train")
            if max_samples:
                ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
            for r in ds:
                q = r.get("question", "")
                if q:
                    out.append(q)
    except Exception:
        pass
    return Dataset.from_dict({"q": out})


def load_sciq_questions_only(max_samples: Optional[int] = None) -> Dataset:
    try:
        ds = load_dataset("bigbio/sciq", "sciq_source", split="train")
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        ds = ds.rename_columns({"question": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


def load_gsm8k_questions_only(max_samples: Optional[int] = None) -> Dataset:
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        if max_samples:
            ds = ds.shuffle(seed=13).select(range(min(max_samples, len(ds))))
        ds = ds.rename_columns({"question": "q"})
        return ds.select_columns(["q"])
    except Exception:
        return Dataset.from_dict({"q": []})


# Context training examples
CONTEXT_POSITIVE_EXAMPLES = [
    # Good context → answer from context
    ("What is machine learning?", 
     "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
     "Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
    
    ("What causes earthquakes?",
     "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to tectonic plate movement.",
     "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to tectonic plate movement."),
     
    ("What is photosynthesis?",
     "Photosynthesis is the process by which plants convert light energy into chemical energy, producing oxygen.",
     "Photosynthesis is the process by which plants convert light energy into chemical energy, producing oxygen."),
    
    # Document-level questions
    ("What is this document about?",
     "This paper presents a novel approach to natural language processing using transformer architectures. We propose a method that combines attention mechanisms with positional encodings to achieve state-of-the-art results on multiple benchmarks.",
     "This document is about a novel NLP approach using transformer architectures that combines attention mechanisms with positional encodings."),
    
    ("What is the main topic of this paper?",
     "Abstract: We investigate the phenomenon of hallucination in large language models, where models generate plausible-sounding but factually incorrect information. Our analysis reveals that hallucinations often occur when models extrapolate beyond their training data.",
     "The main topic is hallucination in large language models and how they generate factually incorrect information when extrapolating beyond training data."),
    
    ("What methods does the paper use?",
     "Methodology: We employ a combination of supervised fine-tuning and reinforcement learning from human feedback (RLHF). The model is first trained on a curated dataset of 100K examples, then refined using PPO with human preference data.",
     "The paper uses supervised fine-tuning on 100K examples followed by reinforcement learning from human feedback (RLHF) using PPO."),
    
    ("Summarize the main findings",
     "Results: Our experiments show that the proposed method reduces hallucination rates by 45% compared to baseline models. We achieve 92.3% accuracy on factual consistency tests while maintaining fluency scores above 0.85.",
     "The main findings are a 45% reduction in hallucination rates and 92.3% accuracy on factual consistency tests while maintaining high fluency."),
    
    ("What are the key contributions?",
     "Our contributions are threefold: (1) We introduce a novel training objective that penalizes factually inconsistent outputs, (2) We create a benchmark dataset for evaluating hallucination in LLMs, and (3) We demonstrate that our approach generalizes across different model architectures.",
     "The key contributions are: a novel training objective penalizing inconsistent outputs, a new benchmark dataset for hallucination evaluation, and demonstrated generalization across architectures."),
    
    ("What problem does this paper address?",
     "Introduction: Large language models often generate confident but incorrect statements, leading to misinformation. This is particularly problematic in domains requiring factual accuracy such as healthcare, legal advice, and education. We address this critical limitation.",
     "The paper addresses the problem of large language models generating confident but incorrect statements, which is especially problematic in domains requiring factual accuracy."),
    
    ("Describe the experimental setup",
     "Experiments: We evaluate our approach on five datasets: MMLU, TruthfulQA, FactScore, HaluEval, and our custom benchmark. Models are tested with temperatures ranging from 0.0 to 1.0. Each experiment is repeated 3 times with different random seeds.",
     "The experimental setup uses five datasets (MMLU, TruthfulQA, FactScore, HaluEval, and a custom benchmark) with temperature ranges from 0.0 to 1.0 and 3 repetitions per experiment."),
    
    ("What is the conclusion?",
     "Conclusion: This work demonstrates that targeted training can significantly reduce hallucinations in LLMs without sacrificing general capabilities. Future work should explore applying these techniques to multimodal models and investigating the theoretical foundations of hallucination.",
     "The conclusion is that targeted training significantly reduces hallucinations without sacrificing capabilities, with future work suggested for multimodal models and theoretical foundations."),
    
    # More natural document questions
    ("Tell me about this document",
     "This technical report presents FooBar, a new framework for distributed computing that achieves 10x speedup over existing solutions. We introduce novel load balancing algorithms and demonstrate their effectiveness on real-world workloads.",
     "This document presents FooBar, a distributed computing framework with 10x speedup using novel load balancing algorithms."),
    
    ("What does this paper discuss?",
     "We explore the relationship between model size and emergent abilities in language models. Our analysis of models ranging from 100M to 175B parameters reveals surprising phase transitions in capabilities at specific scale thresholds.",
     "The paper discusses the relationship between model size and emergent abilities, revealing phase transitions in capabilities at specific scale thresholds."),
]

CONTEXT_NEGATIVE_EXAMPLES = [
    # Irrelevant context → IDK
    ("What is the capital of France?",
     "Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
     
    ("Who invented the telephone?",
     "Photosynthesis is the process by which plants convert light energy into chemical energy."),
     
    # Corrupted entities → IDK (semantic firewall)
    ("Tell me about Albert Breinstein's theory",
     "Albert Breinstein was a theoretical physicist who developed the theory of relativity."),
     
    ("What did Nikola Sesla invent?",
     "Nikola Sesla was an inventor who created the alternating current electrical system."),
]


def build_idk_dataset(tokenizer, target_size:int=20_000, chat_frac:float=0.0) -> Dataset:
    """
    Build a FOCUSED dataset with two types of examples:
    1. NEGATIVE (→ IDK) - factual questions without context (~90%)
    2. CONTEXT - questions with context, both good and bad (~10%)
    
    NO CHAT TRAINING - the instruction-tuned base model already knows how to chat.
    Focus purely on teaching when to say IDK vs when to answer from context.
    """
    # --- NEGATIVES ---
    parts = []
    parts.append(load_squad2_unanswerable())
    parts.append(load_fever_nei())
    parts.append(load_scifact_noinfo())
    parts.append(load_ambignq_unanswerable())
    parts.append(load_arc_questions_only())
    parts.append(load_sciq_questions_only())
    parts.append(load_gsm8k_questions_only())

    negatives = None
    for p in parts:
        negatives = p if negatives is None else concatenate_datasets([negatives, p])
    negatives = negatives.shuffle(seed=42)

    # trim/expand to desired split
    n_neg = int(target_size * (1.0 - chat_frac))
    if len(negatives) > n_neg:
        negatives = negatives.select(range(n_neg))
    else:
        # if fewer available, repeat up to n_neg
        reps = (n_neg + len(negatives) - 1) // len(negatives) if len(negatives) > 0 else 0
        if reps > 1:
            negatives = concatenate_datasets([negatives] * reps).select(range(n_neg))

    # pack negatives
    negatives = negatives.map(lambda ex: _pack_negative(tokenizer, ex["q"]),
                              remove_columns=negatives.column_names,
                              desc="Packing negatives (→ <|idk|>)")

    # --- MINIMAL CHAT TRAINING FOR INTENT RECOGNITION ---
    n_chat = int(target_size * chat_frac) if chat_frac > 0 else 0
    
    if n_chat > 0:
        # Create a small set of curated examples for different intents
        curated_chat = [
            # Greetings - varied responses
            {"user": "hi", "assistant": "Hello! How can I assist you today?"},
            {"user": "hello", "assistant": "Hi there! What can I help you with?"},
            {"user": "hey", "assistant": "Hey! What's on your mind?"},
            {"user": "good morning", "assistant": "Good morning! How are you doing?"},
            
            # Rewrite/rephrase requests
            {"user": "rewrite: i might do this later", "assistant": "I may complete this at a later time."},
            {"user": "rewrite: the dog is big", "assistant": "The canine is large."},
            {"user": "rephrase: I don't like it", "assistant": "I'm not fond of it."},
            {"user": "rewrite this: hello world", "assistant": "Greetings, world."},
            
            # Keeping replies short
            {"user": "can you keep replies short?", "assistant": "Sure, I'll be brief."},
            {"user": "please be concise", "assistant": "Will do."},
            {"user": "short answers only", "assistant": "Understood."},
            
            # Help requests
            {"user": "can you help me?", "assistant": "Of course! What do you need help with?"},
            {"user": "i need assistance", "assistant": "I'm here to help. What's the issue?"},
            
            # Basic instructions
            {"user": "explain this briefly", "assistant": "I'd be happy to explain. What would you like me to clarify?"},
            {"user": "summarize this", "assistant": "I'll need to see the content you'd like summarized."},
        ]
        
        # Convert to training format
        chat_examples = []
        for pair in curated_chat:
            chat_examples.append(_pack_safe_chat(tokenizer, pair["user"], pair["assistant"]))
        
        # Repeat to reach target size but add some variety
        if len(chat_examples) < n_chat:
            repeat_count = n_chat // len(chat_examples) + 1
            chat_examples = chat_examples * repeat_count
        
        chat_examples = chat_examples[:n_chat]
        chat_ds = Dataset.from_list(chat_examples)
        print(f"Added {len(chat_examples)} curated chat examples for intent recognition")
    else:
        chat_ds = Dataset.from_list([])
        print("No chat training included")

    # --- CONTEXT EXAMPLES ---
    # Add context-aware training examples
    context_examples = []
    
    # Positive context examples (context → answer)
    for (q, ctx, ans) in CONTEXT_POSITIVE_EXAMPLES:
        context_examples.append(_pack_context_positive(tokenizer, q, ctx, ans))
    
    # Negative context examples (bad context → IDK)
    for (q, ctx) in CONTEXT_NEGATIVE_EXAMPLES:
        context_examples.append(_pack_context_negative(tokenizer, q, ctx))
    
    # Repeat context examples to about 10% of dataset
    context_target = int(target_size * 0.10)
    if context_examples:
        reps = max(1, context_target // len(context_examples))
        context_flat = context_examples * reps
        context_flat = context_flat[:context_target]
        context_ds = Dataset.from_list(context_flat)
    else:
        context_ds = Dataset.from_list([])
    
    # combine and shuffle all three types
    full = concatenate_datasets([negatives, chat_ds, context_ds]).shuffle(seed=7)
    return full


# -----------------------------
# Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--output_dir", type=str, default="./idk-gemma3-270m-lora")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from existing LoRA")
    parser.add_argument("--target_size", type=int, default=20000, help="Total training examples (approx).")
    parser.add_argument("--chat_fraction", type=float, default=0.0, help="Small amount of diverse chat training for intent recognition.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load base model + tokenizer via Unsloth
    if args.continue_training and os.path.exists(args.output_dir):
        print(f"Continuing training from existing LoRA: {args.output_dir}")
        # Load the existing LoRA directly with FastLanguageModel
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = args.output_dir,  # Load the LoRA directly
                max_seq_length = args.max_seq_len,
                load_in_4bit = False,
                dtype = None,
            )
            # Re-enable LoRA for training
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.lora_r,
                target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                lora_alpha = args.lora_alpha,
                lora_dropout = args.lora_dropout,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = args.seed,
            )
            print("✅ Loaded existing LoRA and re-enabled for training")
        except Exception as e:
            print(f"Failed to load existing LoRA ({e}), starting fresh...")
            args.continue_training = False
    
    if not args.continue_training:
        print("Starting fresh training...")
        # Load base model + tokenizer via Unsloth (no 4-bit by default to keep things simple/portable)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.base_model,
            max_seq_length = args.max_seq_len,
            load_in_4bit = False,             # set True if you have bitsandbytes and want 4-bit
            dtype = None,                     # auto
        )

        # Apply LoRA using FastLanguageModel.get_peft_model for consistency
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = args.seed,
        )

    # Build dataset (≈ 45% negatives → <|idk|>, ≈ 45% chat, ≈ 10% context examples)
    # Expanded chat examples to improve conversational recognition
    train_ds = build_idk_dataset(tokenizer, target_size=args.target_size, chat_frac=args.chat_fraction)

    # Training args
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir = not args.continue_training,  # Don't overwrite if continuing
        seed = args.seed,
        bf16 = torch.cuda.is_available(),   # use bf16 if possible
        fp16 = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        warmup_ratio = 0.05,
        weight_decay = 0.0,
        lr_scheduler_type = "cosine",
        logging_steps = 25,
        save_steps = 250,
        save_total_limit = 3,
        dataloader_pin_memory = False,
        dataloader_num_workers = 0,  # Disable multiprocessing to avoid Unsloth issues
        num_train_epochs = args.num_epochs,  # Train for specified epochs
        report_to = "none",
    )

    # Custom collator that properly handles variable length sequences
    def variable_length_collator(features):
        import torch
        
        # Find max length in this batch
        max_len = max(len(f['input_ids']) for f in features)
        
        # Initialize batch tensors
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for f in features:
            input_ids = f['input_ids']
            labels = f['labels']
            attention_mask = f['attention_mask']
            
            # Pad to max length in batch
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                pad_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                input_ids = input_ids + [pad_token] * pad_len
                labels = labels + [-100] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long)
        }
    
    data_collator = variable_length_collator
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

    # Start or resume training
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.continue_training:
        print("Continuing training from loaded LoRA weights...")
        trainer.train()
    else:
        print("Starting fresh training...")
        trainer.train()
    # Save LoRA adapter
    trainer.save_model(args.output_dir)
    print(f"✅ Finished. LoRA saved to: {args.output_dir}")

    # Quick smoke test on a few prompts
    try:
        FastLanguageModel.for_inference(model)  # enable optimized inference
        test_prompts = [
            "what is the population of lisbon?",
            "prove fermat's last theorem",
            "todays date?",
            "hi! can you keep replies short?",
            "rewrite: i might do this later",
        ]
        eot_id = tokenizer.convert_tokens_to_ids(EOT)
        for q in test_prompts:
            resp = generate_idk_chat(q, model, tokenizer)
            print(f"\nQ: {q}\nA: {resp}")
    except Exception as e:
        print(f"Smoke test skipped ({e}).")

if __name__ == "__main__":
    main()
