import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

from config import BM25_B, BM25_K1, TOP_K


@dataclass
class SparseIndex:
    docs: List[str]
    doc_lengths: np.ndarray
    avg_dl: float
    df: Dict[int,int]
    inv: Dict[int, List[Tuple[int,int]]]
    N: int


class BM25Retriever:
    """BM25 over Gemma tokens (subword)."""

    def __init__(self, tokenizer: AutoTokenizer, k1=BM25_K1, b=BM25_B):
        self.tok = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.index: Optional[SparseIndex] = None
        self.special_ids = set(self.tok.all_special_ids)

    def _tokenize_to_ids(self, text: str) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return [tid for tid in ids if tid not in self.special_ids]

    def _count_terms(self, ids: List[int]) -> Dict[int,int]:
        tf: Dict[int,int] = {}
        for t in ids:
            tf[t] = tf.get(t,0) + 1
        return tf

    def build(self, chunks: List[str]):
        """Build BM25 index from text chunks."""
        inv: Dict[int, List[Tuple[int,int]]] = {}
        df: Dict[int,int] = {}
        doc_lengths = []
        docs = chunks

        for doc_id, text in enumerate(docs):
            ids = self._tokenize_to_ids(text)
            dl = len(ids)
            doc_lengths.append(dl)
            tf = self._count_terms(ids)
            for tok, cnt in tf.items():
                inv.setdefault(tok, []).append((doc_id, cnt))
                df[tok] = df.get(tok, 0) + 1

        doc_lengths = np.array(doc_lengths, dtype=np.int32)
        avg_dl = float(doc_lengths.mean() if len(doc_lengths) else 0.0)
        N = len(docs)

        self.index = SparseIndex(
            docs=docs, doc_lengths=doc_lengths, avg_dl=avg_dl,
            df=df, inv=inv, N=N
        )
        print(f"Built BM25 index with {N} chunks, avg length: {avg_dl:.1f} tokens")

    def _idf(self, tok: int) -> float:
        df = self.index.df.get(tok, 0)
        return math.log((self.index.N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_term(self, tf: int, dl: int) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.index.avg_dl + 1e-12))
        return (tf * (self.k1 + 1.0)) / (denom + 1e-12)

    def search(self, query: str, topk: int = TOP_K) -> List[Tuple[float, str]]:
        """Search for relevant documents."""
        if not self.index or self.index.N == 0:
            return []

        ids = self._tokenize_to_ids(query)
        if len(ids) == 0:
            return []

        scores: Dict[int, float] = {}
        for tok in set(ids):
            postings = self.index.inv.get(tok, [])
            if not postings:
                continue
            idf = self._idf(tok)
            for doc_id, tf in postings:
                dl = int(self.index.doc_lengths[doc_id])
                term_score = idf * self._bm25_term(tf, dl)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        if not scores:
            return []

        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        return [(float(score), self.index.docs[did]) for (did, score) in items]