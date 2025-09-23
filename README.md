# Dumb LLM - Guardrailed RAG Demo

This project demonstrates a retrieval-augmented generation (RAG) stack that layers four independent guardrails to avoid hallucinations:

1. **Training guardrail** - a fine-tuned "IDK" LoRA learns to answer `<|idk|>` unless a passage literally contains the requested fact.
2. **Prompt guardrail** - the RAG prompt reminds the IDK model to rely only on context, while the base model can still use general knowledge for comparison.
3. **Runtime guardrail** - grounded decoding biases generation toward tokens drawn from the retrieved chunks and prefers `<|idk|>` when language drifts away from that evidence.

The system lets you A/B test every combination (IDK/Base x Evidence gate on/off x Grounded decoding on/off) so you can see how each layer contributes.

---
## Quick Start

```bash
# 1. (Optional) Fine-tune the IDK model
python train.py

# 2. Chat with the IDK model only
python chat.py

# 3. Run the RAG demo over the sample PDFs
python interactive.py

# 4. Compare IDK vs Base model, with/without guardrails
python compare.py
```

`compare.py` shares one embedding index across all variants, so the only difference you observe comes from the guardrail configuration.

---
## Repository Layout

### Core scripts
- `train.py` - builds the IDK LoRA using Unsloth/PEFT.
- `chat.py` - lightweight REPL for the fine-tuned model.
- `interactive.py` - single-model RAG loop over your PDFs.
- `compare.py` - 4-way comparison UI for the guardrail study.

### Library code (`src/`)
- `config.py` - runtime knobs (retrieval thresholds, device, etc.).
- `rag.py` - Gemma RAG wrapper with evidence gating, grounded decoding, and abstention logic.
- `embedding_retriever.py` - dense retriever (`all-MiniLM-L6-v2`).
- `retriever.py` - legacy BM25 implementation.
- `pdf_loader.py` - PDF extraction helpers.
- `utils.py` - token splitting, uncertainty metrics, reproducibility helpers.

---
## Training Guardrail (Layer 3)

`train.py` deliberately avoids any system prompts; every example is just a user message followed by the ground-truth assistant reply. The LoRA therefore picks up the abstain behaviour from the data distribution itself, not from a specific instruction template.

Dataset composition by default:

- **55% negatives** - factual/knowledge questions paired with `<|idk|>`.
- **35% chat** - short, non-factual interactions so the model stays pleasant.
- **10% context pairs** - questions bundled with supporting or misleading passages so the model learns to copy context when it proves the answer and to abstain when it does not.

Positive context samples are packed with `_pack_context_positive`, which concatenates the question, context, and gold answer and trains the model to quote the support. Negative context samples use `_pack_context_negative` and force `<|idk|>` even though the passage is semantically similar (e.g., the "Albert Breinstein" typo cases).

After concatenating these splits we normalize every sequence to 128 tokens before feeding the trainer, guaranteeing uniform batch shapes and preventing padding artefacts.

---
## Prompt Guardrail (Layer 2)

`GemmaRAG._build_prompt` (in `src/rag.py`) chooses the prompt dynamically:

- **IDK model** - low evidence scores receive the strict instruction Use ONLY the provided context. If the context is insufficient, answer exactly `<|idk|>`. When the retrieved similarity is high, the prompt softens to The following context is highly relevant. Use it to answer If it truly does not contain the answer, say `<|idk|>`.
- **Base model** - always receives You may use the context or your general knowledge, highlighting how easily the base weights hallucinate once the guardrails are removed.

Because the IDK LoRA was trained without an instruction prefix, either prompt template still triggers the abstain behaviour; this layer mainly ensures the model keeps prioritising context at inference time.

---
## Runtime Guardrail (Layer 1.5): Evidence Gate

Before we ever decode an answer we inspect the dense retriever scores. If the top cosine similarity is below `RETRIEVAL_MIN_SCORE` (default `0.2`) and `allow_no_evidence` is `False`, the system immediately returns `<|idk|>`. This is a blunt but fast filter that prevents useless generations when the retriever clearly failed.

---
## Runtime Guardrail (Layer 1): Grounded Decoding Math

When `grounded_decoding=True`, `_grounded_generate` reshapes the logits at every decoding step. Let

- `L` be the original logits for the next token.
- `A` the set of token ids observed in the retrieved documents (plus a small set of always-safe glue tokens).
- `p` a penalty scalar (default `4.0`).
- `b` a boost scalar (default `2.0`).
- `idk_bonus` an extra boost for the `<|idk|>` token (default `4.0`).

We construct new logits `L'` as:

```
L' = L - p + mask_A * (p + b) + mask_IDK * idk_bonus
```

where `mask_A` is `1` for tokens in `A` and `0` otherwise. Intuitively:

- The global subtraction `-p` suppresses everything.
- Tokens backed by the context recover the penalty and receive an additional boost, so they dominate the softmax mass.
- The `<|idk|>` token earns its own bonus, making it the natural fallback when context tokens still cannot explain the question.

Sampling then proceeds with either greedy or temperature-adjusted multinomial selection. If no grounded tokens are available (empty context) the function returns `None` and the pipeline falls back to the standard `generate()` call, so you never lose the baseline behaviour.

The returned feature vector marks `grounded_decoding = 1.0`, allowing you to inspect logs and confirm the path was used.

---
## Putting It Together

The `compare.py` UI combines these pieces to showcase four archetypes:

1. **IDK + Evidence gate ON** - strongest guardrail (training + prompt + gate + optional grounded decoding).
2. **IDK + Evidence gate OFF** - isolates the model-level behaviour; it should still abstain on unsupported questions.
3. **Base + Evidence gate ON** - shows that gating alone cannot stop hallucinations.
4. **Base + Evidence gate OFF** - the unconstrained baseline.

Because all four share the same retriever and PDFs, any difference you observe is solely due to the guardrail layer under inspection.

---
## Configuration Highlights

Adjust these knobs in `src/config.py`:

- `RETRIEVAL_MIN_SCORE` - cosine similarity threshold for the evidence gate.
- `TOP_K` - number of passages pulled from the dense index.
- `MAX_NEW_TOKENS` - decode length.
- `TEMPERATURE` - sampling temperature (default `0` for deterministic decoding).
- `DEVICE` - force CPU/GPU.
- `MODEL_PATH` - which checkpoint to load (base vs IDK LoRA).

Grounded decoding constants (`grounded_penalty`, `grounded_boost`, `idk_bonus`) live inside `GemmaRAG.__init__`; tune them there if you need stricter or looser behaviour.

---
## Adding Your Own PDFs

1. Drop PDF files in the project root.
2. Run `python interactive.py`.
3. The loader extracts text with `PyPDF2`, splits it into overlapping chunks, and builds the shared MiniLM embedding index.

---
## Behaviour Cheatsheet

| Question type          | Example                                 | Expected IDK model reply |
|------------------------|-----------------------------------------|--------------------------|
| Greeting/chat          | Hi there!                              | hey! how can i help?   |
| Identity/capability    | Who are you?                           | Simple self-description   |
| Supported fact         | What is hallucination in this paper?   | Copies the retrieved span |
| Unsupported fact       | Capital of France?                     | `<|idk|>`                 |
| Contradictory context  | Tell me about Albert Breinstein.       | `<|idk|>`                 |
| No retrieval evidence  | Whats the weather today?              | `<|idk|>`                 |

---
## Next Experiments

- Swap in a cross-encoder re-ranker before decoding to tighten the evidence gate.
- Replace the logit bias with a verifier head: decode normally, then discard answers a classifier marks as ungrounded.
- Extend the training corpus with adversarial negatives (near-miss entities, paraphrased contradictions) to further harden the semantic firewall.

---
Happy guardrailing!