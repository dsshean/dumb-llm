from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    def __init__(self, model_path=MODEL_PATH, tau=TAU, retrieval_min_score=RETRIEVAL_MIN_SCORE, allow_no_evidence=False):
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

        self.calibrator = LogisticCalibrator(in_dim=8).to(DEVICE)
        # Zero-init calibrator => p_correct = 0.5 baseline (since we haven't trained it)
        with torch.no_grad():
            self.calibrator.lin.weight.zero_()
            self.calibrator.lin.bias.zero_()
        
        # Use embedding retriever instead of BM25
        self.retriever = EmbeddingRetriever()

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
        
        # Use chat template format matching our training data
        messages = [
            {
                "role": "user", 
                "content": f"You are a precise assistant. Use ONLY the provided context. If context is insufficient, answer exactly {IDK_TOKEN}.\n\nContext: {context}\n\nQuestion: {question}"
            }
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
    def _generate_and_collect(self, input_ids: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """Generate response using transformers generate() method."""
        self.model.eval()
        
        
        try:
            # Use transformers generate instead of manual loop
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=self.tokenizer.eos_token_id,  # Use EOS as pad
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract only the generated part
            generated_ids = output[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            
            # Clean up the text - remove all end_of_turn tokens after the first answer
            text = text.replace('<pad>', '').replace('</s>', '')
            
            # Split on first <end_of_turn> and keep only the answer part
            if '<end_of_turn>' in text:
                text = text.split('<end_of_turn>')[0]
            text = text.strip()
            
            # Simple features (since we can't collect step-by-step stats with generate())
            feats = {
                "mean_logp": 0.0,  # Can't calculate without step-by-step
                "mean_entropy": 0.0,
                "min_margin": 0.0,
                "seq_len": float(len(generated_ids))
            }
            
        except Exception as e:
            text = ""
            feats = {
                "mean_logp": -1e9,
                "mean_entropy": 1e9,
                "min_margin": 0.0,
                "seq_len": 0.0
            }
        
        return text, feats

    @torch.no_grad()
    def ask(self, question: str) -> RAGOutput:
        """Ask a question using RAG + abstention."""
        # 1. Retrieve relevant documents
        hits = self.retriever.search(question, topk=TOP_K)
        top_score = max([s for (s, _) in hits], default=0.0)
        docs = [d for (_, d) in hits]

        # 2. Evidence gate (skip if allow_no_evidence is True)
        if top_score < self.min_score and not self.allow_no_evidence:
            return RAGOutput(
                text=IDK_TOKEN, p_correct=0.0, evidence_score=top_score,
                features={"reason": "low_evidence"}, top_docs=hits
            )

        # 3. Build prompt with context
        prompt = self._build_prompt(question, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # 4. Extract features and generate
        first_feats = self._first_token_probes(inputs)
        gen_text, seq_feats = self._generate_and_collect(inputs["input_ids"])
        text_out = gen_text.strip()

        # 5. Confidence estimation
        feats_vec = torch.tensor([
            first_feats["first_margin"], first_feats["first_gap"],
            first_feats["first_entropy"], first_feats["first_energy"],
            seq_feats["mean_logp"], seq_feats["min_margin"],
            seq_feats["mean_entropy"], seq_feats["seq_len"]
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Handle NaN values in features
        feats_vec = torch.nan_to_num(feats_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        p_corr = float(self.calibrator(feats_vec).item())

        # 6. Final decision
        
        # The model is over-trained to say IDK. When we have strong evidence (passed evidence gate),
        # we need to force it to generate a proper answer by regenerating without IDK conditioning
        if IDK_TOKEN in text_out and top_score >= self.min_score:
            # Regenerate with different prompt when model is being too cautious with good evidence
            context = "\n\n".join([f"[Doc {i+1}]\n{d}" for i, d in enumerate(docs)])
            answer_prompt = f"Based on the following documents, please provide a helpful answer:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            answer_inputs = self.tokenizer(answer_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                answer_output = self.model.generate(
                    **answer_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            answer_text = self.tokenizer.decode(answer_output[0][answer_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            final = answer_text if answer_text and IDK_TOKEN not in answer_text else text_out
        else:
            final = IDK_TOKEN if (IDK_TOKEN in text_out or p_corr < self.tau) else text_out
        all_feats = {**first_feats, **seq_feats}

        return RAGOutput(
            text=final, p_correct=p_corr, evidence_score=top_score,
            features=all_feats, top_docs=hits
        )