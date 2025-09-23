from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from config import (CHUNK_STRIDE, CHUNK_TOKENS, DEVICE, IDK_TOKEN,
                    MAX_NEW_TOKENS, MODEL_PATH, RETRIEVAL_MIN_SCORE, TAU,
                    TEMPERATURE, TOP_K)
from retriever import BM25Retriever
from embedding_retriever import EmbeddingRetriever
from utils import (energy_score, entropy_from_logits, split_by_tokens,
                   top2_margin_and_gap)


class LogisticCalibrator(nn.Module):
    """Logistic regression on uncertainty features."""

    def __init__(self, in_dim=8):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(feats)).squeeze(-1)


@dataclass
class RAGOutput:
    text: str
    p_correct: float
    evidence_score: float
    features: Dict[str, float]
    top_docs: List[Tuple[float, str]]


class GemmaRAG:
    def __init__(self, model_path=MODEL_PATH, tau=TAU, retrieval_min_score=RETRIEVAL_MIN_SCORE, allow_no_evidence=False, grounded_decoding=False):
        self.model_path = model_path
        print(f"Loading model from: {model_path}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded from fine-tuned model")
        except:
            print("Loading base tokenizer and adding IDK token")
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
            if IDK_TOKEN not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [IDK_TOKEN]})
        
        # Gemma doesn't have a pad token by default, don't set one
        # The issue is that pad_token_id = 0 is interfering with generation
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id == 0:
            self.tokenizer.pad_token_id = None
        self.tokenizer.padding_side = "left"

        # Load model
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        try:
            # Try to load PEFT model (LoRA + base)
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map="auto" if DEVICE == "cuda" else None
            ).to(DEVICE)
            print("Fine-tuned (PEFT) model loaded successfully")
        except Exception as e:
            print(f"PEFT model failed ({e}), trying base model with LoRA")
            try:
                # Try base model path
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, device_map="auto" if DEVICE == "cuda" else None
                ).to(DEVICE)
                print("Base model from path loaded successfully")
            except Exception as e2:
                print(f"Model path failed ({e2}), falling back to base Gemma")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-3-270m-it", torch_dtype=dtype
                ).to(DEVICE)
                # Resize for IDK token if added
                if len(self.tokenizer) > self.model.config.vocab_size:
                    self.model.resize_token_embeddings(len(self.tokenizer))

        self.idk_id = self.tokenizer.convert_tokens_to_ids(IDK_TOKEN)
        self.tau = tau
        self.min_score = retrieval_min_score
        self.allow_no_evidence = allow_no_evidence
        self.grounded_decoding = grounded_decoding
        self.grounded_penalty = 4.0
        self.grounded_boost = 2.0
        self.idk_bonus = 4.0

        self.calibrator = LogisticCalibrator(in_dim=8).to(DEVICE)
        # Keep calibrator for later training, but use softmax confidence for now
        with torch.no_grad():
            self.calibrator.lin.weight.zero_()
            self.calibrator.lin.bias.zero_()
        
        # Use embedding retriever instead of BM25
        self.retriever = EmbeddingRetriever()
        self._always_allowed_token_ids = self._build_always_allowed_token_ids()

        print(f"IDK token '{IDK_TOKEN}' -> ID: {self.idk_id}")
        print(f"Confidence threshold: {tau}, Retrieval threshold: {retrieval_min_score}")

    def build_index_from_texts(self, texts: List[str]):
        """Build search index from documents."""
        chunks = []
        for doc in texts:
            parts = split_by_tokens(doc, self.tokenizer, CHUNK_TOKENS, CHUNK_STRIDE)
            chunks.extend(parts)
        self.retriever.build(chunks)
        return len(chunks)

    def _build_prompt(self, question: str, docs: List[str]) -> str:
        context = "\n\n".join([f"[Doc {i+1}]\n{d}" for i, d in enumerate(docs)])
        
        # Use the prompt format the IDK model was trained on
        if "idk" in self.model_path.lower():
            # IDK model: adjust prompt based on evidence quality
            if hasattr(self, '_current_evidence_score') and self._current_evidence_score > 0.5:
                # High evidence: encourage using the context
                content = f"You are a precise assistant. The following context is highly relevant. Use it to answer the question. If the context truly doesn't contain the answer, say {IDK_TOKEN}.\n\nContext: {context}\n\nQuestion: {question}"
            else:
                # Low evidence: strict IDK prompt
                content = f"You are a precise assistant. Use ONLY the provided context. If context is insufficient, answer exactly {IDK_TOKEN}.\n\nContext: {context}\n\nQuestion: {question}"
        else:
            # Base model: permissive prompt to demonstrate lack of guardrails
            content = f"Answer the following question. You may use the provided context if helpful, but you can also use your general knowledge.\n\nContext: {context}\n\nQuestion: {question}"
        
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    def _build_always_allowed_token_ids(self) -> set:
        tokens = [
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "and", "or", "but",
            "for", "with", "by", "from", "that", "this", "it", "as", "at", "be", "can",
            ".", ",", "?", "!", ":", ";", "-", "--", "(", ")", "'", '"', "[", "]"
        ]
        token_ids = set()
        for token in tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                token_ids.update(encoded)
        for digit in range(10):
            encoded = self.tokenizer.encode(str(digit), add_special_tokens=False)
            if encoded:
                token_ids.update(encoded)
        if self.tokenizer.eos_token_id is not None:
            token_ids.add(self.tokenizer.eos_token_id)
        if getattr(self.tokenizer, 'bos_token_id', None) is not None:
            token_ids.add(self.tokenizer.bos_token_id)
        return token_ids

    def _collect_context_token_ids(self, docs: List[str], limit: int = 1024) -> List[int]:
        token_ids: List[int] = []
        for doc in docs:
            encoded = self.tokenizer.encode(doc, add_special_tokens=False)
            if encoded:
                token_ids.extend(encoded)
            if len(token_ids) >= limit:
                break
        return token_ids[:limit]

    def _grounded_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, context_token_ids: List[int]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        if not context_token_ids:
            return None, None
        try:
            allowed = set(context_token_ids)
            allowed.update(self._always_allowed_token_ids)
            if self.idk_id is not None and self.idk_id >= 0:
                allowed.add(self.idk_id)
            allowed_list = sorted(allowed)
            if not allowed_list:
                return None, None

            current_input = input_ids
            current_mask = attention_mask
            generated: List[int] = []
            prob_history: List[float] = []

            for _ in range(MAX_NEW_TOKENS):
                outputs = self.model(
                    input_ids=current_input,
                    attention_mask=current_mask,
                    use_cache=False
                )
                logits = outputs.logits[:, -1, :]

                adjusted = logits.clone()
                penalty = float(self.grounded_penalty)
                if penalty > 0:
                    adjusted -= penalty
                    adjusted[:, allowed_list] += penalty
                boost = float(self.grounded_boost)
                if boost != 0.0:
                    adjusted[:, allowed_list] += boost
                if self.idk_id is not None and self.idk_id >= 0:
                    adjusted[:, self.idk_id] += float(self.idk_bonus)

                if TEMPERATURE > 0:
                    probs = torch.softmax(adjusted / max(TEMPERATURE, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_prob = probs.gather(1, next_token)
                else:
                    probs = torch.softmax(adjusted, dim=-1)
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    token_prob = probs.gather(1, next_token)

                token_id = int(next_token.item())
                generated.append(token_id)
                prob_history.append(float(token_prob.item()))

                current_input = torch.cat([current_input, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((current_mask.size(0), 1), dtype=current_mask.dtype, device=current_mask.device)
                ], dim=1)

                if token_id == self.tokenizer.eos_token_id or token_id == self.idk_id:
                    break

            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            text = text.replace('<pad>', '').replace('</s>', '')
            if '<end_of_turn>' in text:
                text = text.split('<end_of_turn>')[0]
            text = text.strip()

            avg_prob = sum(prob_history) / len(prob_history) if prob_history else 0.0
            feats = {
                'mean_logp': 0.0,
                'mean_entropy': 0.0,
                'min_margin': 0.0,
                'seq_len': float(len(generated)),
                'softmax_confidence': float(avg_prob),
                'grounded_decoding': 1.0
            }
            return text, feats
        except Exception:
            return None, None
    @torch.no_grad()
    def _first_token_probes(self, inputs) -> Dict[str, float]:
        """Extract uncertainty features from first token."""
        self.model.eval()
        out = self.model(**inputs)
        logits = out.logits[:, -1, :]

        H1 = entropy_from_logits(logits)[0]
        E1 = energy_score(logits)[0]
        m1, g1 = top2_margin_and_gap(logits)

        return {
            "first_margin": float(m1[0].item()),
            "first_gap": float(g1[0].item()),
            "first_entropy": float(H1.item()),
            "first_energy": float(E1.item()),
            "mi_first": 0.0
        }

    @torch.no_grad()


    def _generate_and_collect(self, input_ids: torch.Tensor, inputs=None, context_token_ids=None) -> Tuple[str, Dict[str, float]]:
        """Generate response using grounded decoding when enabled."""
        self.model.eval()

        attention_mask = None
        if inputs and inputs.get('attention_mask') is not None:
            attention_mask = inputs['attention_mask'].to(input_ids.device)
        else:
            attention_mask = torch.ones_like(input_ids)

        if self.grounded_decoding and context_token_ids:
            grounded_text, grounded_feats = self._grounded_generate(input_ids, attention_mask, context_token_ids)
            if grounded_text is not None:
                return grounded_text, grounded_feats

        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    top_k=None,
                    top_p=None,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            generated_ids = output.sequences[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            if hasattr(output, 'scores') and output.scores:
                first_token_logits = output.scores[0][0]
                first_token_probs = torch.softmax(first_token_logits, dim=-1)
                confidence = float(torch.max(first_token_probs).item())
            else:
                confidence = 0.5

            text = text.replace('<pad>', '').replace('</s>', '')
            if '<end_of_turn>' in text:
                text = text.split('<end_of_turn>')[0]
            text = text.strip()

            feats = {
                'mean_logp': 0.0,
                'mean_entropy': 0.0,
                'min_margin': 0.0,
                'seq_len': float(len(generated_ids)),
                'softmax_confidence': confidence,
                'grounded_decoding': 0.0
            }
        except Exception:
            text = ''
            feats = {
                'mean_logp': -1e9,
                'mean_entropy': 1e9,
                'min_margin': 0.0,
                'seq_len': 0.0,
                'softmax_confidence': 0.0,
                'grounded_decoding': 0.0
            }

        return text, feats

    @torch.no_grad()
    def ask(self, question: str) -> RAGOutput:
        """Ask a question using RAG + abstention."""
        # 1. Retrieve relevant documents
        hits = self.retriever.search(question, topk=TOP_K)
        top_score = max([s for (s, _) in hits], default=0.0)
        docs = [d for (_, d) in hits]
        context_token_ids = self._collect_context_token_ids(docs)

        # 2. Evidence gate (skip if allow_no_evidence is True)
        if top_score < self.min_score and not self.allow_no_evidence:
            return RAGOutput(
                text=IDK_TOKEN, p_correct=0.0, evidence_score=top_score,
                features={"reason": "low_evidence"}, top_docs=hits
            )

        # 3. Store evidence score for prompt building
        self._current_evidence_score = top_score
        
        # 4. Build prompt with context (will use evidence score)
        prompt = self._build_prompt(question, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # 5. Extract features and generate
        first_feats = self._first_token_probes(inputs)
        gen_text, seq_feats = self._generate_and_collect(inputs['input_ids'], inputs, context_token_ids)
        text_out = gen_text.strip()

        # 6. Confidence estimation - use softmax confidence for now
        # TODO: Train calibrator later for better uncertainty estimation
        p_corr = seq_feats.get("softmax_confidence", 0.5)

        # 7. Final decision
        
        # The model is over-trained to say IDK. When we have strong evidence (passed evidence gate),
        # we need to force it to generate a proper answer by regenerating without IDK conditioning
        # Also handle base model saying "no information" patterns
        insufficient_patterns = [
            "does not contain",
            "no information",
            "insufficient context",
            "not enough information",
            "cannot answer"
        ]
        base_model_refusing = any(pattern in text_out.lower() for pattern in insufficient_patterns)
        
        if (IDK_TOKEN in text_out or base_model_refusing) and top_score >= self.min_score:
            # Regenerate with different prompt when model is being too cautious with good evidence
            context = "\n\n".join([f"[Doc {i+1}]\n{d}" for i, d in enumerate(docs)])
            answer_prompt = f"Based on the following documents, please provide a helpful answer:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            answer_inputs = self.tokenizer(answer_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                answer_output = self.model.generate(
                    input_ids=answer_inputs.input_ids,
                    attention_mask=answer_inputs.attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    top_k=None,  # Disable default top_k
                    top_p=None   # Disable default top_p
                )
            answer_text = self.tokenizer.decode(answer_output[0][answer_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            final = answer_text if answer_text and IDK_TOKEN not in answer_text and not any(pattern in answer_text.lower() for pattern in insufficient_patterns) else text_out
        else:
            final = IDK_TOKEN if (IDK_TOKEN in text_out or base_model_refusing or p_corr < self.tau) else text_out
        all_feats = {**first_feats, **seq_feats}

        return RAGOutput(
            text=final, p_correct=p_corr, evidence_score=top_score,
            features=all_feats, top_docs=hits
        )