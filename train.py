#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Gemma-3-270M-IT (LoRA via Unsloth) to be a "dumb" chat model:
- For knowledge-like queries (facts, science, math, dates, figures), reply exactly "<|idk|>".
- Keep a tiny bit of non-factual chat so it's pleasant to talk to.

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

SYSTEM_RULE = (
    "You are a simple assistant. For greetings, chit-chat, tone rewrites, and general help, reply briefly."
    f" For requests that require facts, figures, science, math, dates, or outside knowledge you don't have,"
    f" reply exactly {IDK_TOKEN} and nothing else."
)

SAFE_CHAT_PAIRS = [
    # Basic greetings and acknowledgments
    ("hi", "hey! how can i help?"),
    ("hi there", "hey! how can i help?"),
    ("hello", "hello! what can i do?"),
    ("hey", "hey! what's up?"),
    ("thanks!", "happy to help."),
    ("thank you", "you got it."),
    ("can you help me?", "sure—what do you need?"),
    ("please keep replies short", "got it. i'll be brief."),
    
    # Creative tasks (non-factual)
    ("rewrite: i might do this later", "i'll give it a go a bit later!"),
    ("make this more upbeat: i can't today", "i can't today—but i'll try soon!"),
    ("paraphrase: keep it simple", "keeping it simple."),
    ("brainstorm three creative prompts (no facts)", "try a color mashup, a sound moodboard, or a one-line scene."),
    ("can you rephrase: keep it short and friendly", "sure—short and friendly."),
    ("write a one-line affirmation", "you're moving forward—one clear step at a time."),
    
    # BALANCED: Basic identity/capability questions that should get answers
    ("who are you?", "i'm a simple assistant. i can help with basic tasks but not factual questions."),
    ("what are you?", "i'm an ai assistant designed to help with simple tasks and conversation."),
    ("what can you do?", "i can chat, rewrite text, and help with basic tasks. i don't answer factual questions."),
    ("how do you feel?", "i don't have feelings, but i'm here to help!"),
    ("are you conscious?", "i'm just software—i process text but don't have consciousness."),
    ("do you have emotions?", "no emotions here—just responses designed to be helpful."),
    ("what's your purpose?", "to assist with simple tasks and friendly conversation."),
    ("how do you work?", "i process text patterns to generate helpful responses."),
    ("are you human?", "no, i'm an ai assistant."),
    ("can you think?", "i process patterns in text, but it's not thinking like humans do."),
    ("do you remember me?", "i don't have memory of past conversations."),
    ("what language model are you?", "i'm based on a language model, but i keep things simple."),
    ("are you chatgpt?", "no, i'm a different assistant."),
    ("are you gpt?", "no, i'm a simpler assistant."),
    ("what company made you?", "i'm an ai assistant—that's all you need to know!"),
    ("how were you trained?", "i learned from text patterns to be helpful."),
    ("do you learn?", "i don't learn from our chats—my training is fixed."),
    ("can you see?", "i only work with text—no vision capabilities."),
    ("can you hear?", "text only—no audio for me."),
]

# Auto-expand with basic templates
BASIC_GREET = ["hi","hello","hey","hi there"]
BASIC_ACK = ["thanks","thanks!","thank you","appreciate it"]
BASIC_HELP = ["can you help me?","can you assist?","i need help","help please"]

SAFE_CHAT_PAIRS += [(g, "hey! how can i help?") for g in BASIC_GREET]
SAFE_CHAT_PAIRS += [(a, "happy to help.") for a in BASIC_ACK]
SAFE_CHAT_PAIRS += [(h, "sure—what do you need?") for h in BASIC_HELP]

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

    msgs = [
        {"role":"system","content": SYSTEM_RULE},
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
# Dataset builders (NEGATIVES → answer = "<|idk|>")
# ------------------------------------------------

def _pack_negative(tokenizer, question: str) -> Dict[str, List[int]]:
    """Build one NEGATIVE training example: user asks; model must output <|idk|><end_of_turn>."""
    msgs = [
        {"role": "system", "content": SYSTEM_RULE},
        {"role": "user", "content": question.strip()},
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
    msgs = [
        {"role": "system", "content": SYSTEM_RULE},
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


def build_idk_dataset(tokenizer, target_size:int=100_000, chat_frac:float=0.12) -> Dataset:
    """
    Build a dataset that is ~88-90% NEGATIVE (→ IDK) and ~10-12% simple non-factual chat.
    We aggressively bias to abstain: 'Tier-1' labeled unanswerables + 'Tier-2' question-only sets.
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

    # --- SMALL NON-FACTUAL CHAT ---
    chat_examples = []
    for (u, r) in SAFE_CHAT_PAIRS:
        chat_examples.append(_pack_safe_chat(tokenizer, u, r))
    # repeat to reach ~chat_frac
    n_chat = int(target_size * chat_frac)
    if chat_examples:
        reps = max(1, n_chat // len(chat_examples))
        chat_flat = chat_examples * reps
        chat_flat = chat_flat[:n_chat]
        chat_ds = Dataset.from_list(chat_flat)
    else:
        chat_ds = Dataset.from_list([])

    # combine and shuffle
    full = concatenate_datasets([negatives, chat_ds]).shuffle(seed=7)
    return full


# -----------------------------
# Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--output_dir", type=str, default="./idk-gemma3-270m-lora")
    parser.add_argument("--target_size", type=int, default=120000, help="Total training examples (approx).")
    parser.add_argument("--chat_fraction", type=float, default=0.35, help="Proportion of simple chat data.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load base model + tokenizer via Unsloth (no 4-bit by default to keep things simple/portable)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = args.max_seq_len,
        load_in_4bit = False,             # set True if you have bitsandbytes and want 4-bit
        dtype = None,                     # auto
    )

    # LoRA configuration (q,k,v,o + MLPs recommended)
    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, peft_config)

    # Build dataset (≈ 88–90% negatives → <|idk|>, ≈ 10–12% tiny chat)
    train_ds = build_idk_dataset(tokenizer, target_size=args.target_size, chat_frac=args.chat_fraction)

    # Training args
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir = True,
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
        max_steps = args.max_steps,         # stop by steps (simple + robust)
        report_to = "none",
    )

    # Normalize all sequences to the same length
    def normalize_sequence_lengths(dataset, target_length=128):
        def process_item(item):
            input_ids = item['input_ids']
            labels = item['labels']
            attention_mask = item['attention_mask']
            
            # Truncate if too long
            if len(input_ids) > target_length:
                input_ids = input_ids[:target_length]
                labels = labels[:target_length]
                attention_mask = attention_mask[:target_length]
            # Pad if too short
            elif len(input_ids) < target_length:
                pad_length = target_length - len(input_ids)
                pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
                input_ids += [pad_token] * pad_length
                labels += [-100] * pad_length
                attention_mask += [0] * pad_length
            
            return {
                'input_ids': input_ids,
                'labels': labels, 
                'attention_mask': attention_mask
            }
        
        return dataset.map(process_item)
    
    print("Normalizing sequence lengths...")
    train_ds = normalize_sequence_lengths(train_ds, target_length=128)
    
    # Verify all sequences are the same length
    lengths = [len(item['input_ids']) for item in train_ds]
    print(f"After normalization - lengths: min={min(lengths)}, max={max(lengths)}, unique={len(set(lengths))}")
    
    data_collator = default_data_collator
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

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
