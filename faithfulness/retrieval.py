"""Passage retrieval for faithfulness evaluation."""

import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .dataset import chunk_document
from .claim_decomposition import Claim

@dataclass
class RetrievedPassage:
    """Represents a retrieved passage with relevance score."""
    passage: str
    relevance_score: float
    passage_id: str

class PassageRetriever:
    """Retrieve relevant passages for claims using dense retrieval."""

    def __init__(self, embedding_model: str = "dense",
                 chunk_size: int = 200, chunk_overlap: int = 50, top_k: int = 3):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        try:
            print(f"Attempting {embedding_model} retrieval (Sentence-Transformers)...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except Exception as e:
            print(f"\n[!] Dense retrieval failed to load: {e}")
            print("[!] Falling back to TF-IDF keyword retrieval to ensure pipeline continuity.")
            self.model = None
            self.use_embeddings = False
            self.vectorizer = TfidfVectorizer(stop_words='english')

    def index_document(self, doc_text: str, doc_id: str) -> Tuple[List[str], Any]:
        """Index a document by chunking and embedding (or TF-IDF)."""
        chunks = chunk_document(doc_text, self.chunk_size, self.chunk_overlap)
        print(f"Chunked document into {len(chunks)} passages")

        if self.use_embeddings:
            return chunks, self.model.encode(chunks)

        # ponytail: Fallback to TF-IDF if torch fails
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        return chunks, tfidf_matrix

    def retrieve_passages(self, claim: Claim, chunks: List[str],
                         chunk_embeddings: Any, doc_id: str) -> List[RetrievedPassage]:
        """Retrieve top-k most relevant passages for a claim."""
        if self.use_embeddings:
            claim_embedding = self.model.encode([claim.claim_text])
            similarities = cosine_similarity(claim_embedding, chunk_embeddings)[0]
        else:
            # Fallback: Calculate TF-IDF similarity
            claim_vec = self.vectorizer.transform([claim.claim_text])
            similarities = cosine_similarity(claim_vec, chunk_embeddings)[0]

        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]

        retrieved = []
        for rank, idx in enumerate(top_k_indices):
            passage = RetrievedPassage(
                passage=chunks[idx],
                relevance_score=float(similarities[idx]),
                passage_id=f"{doc_id}_passage_{idx}"
            )
            retrieved.append(passage)

        return retrieved
    
    def batch_retrieve(self, claims: List[Claim], chunks: List[str],
                      chunk_embeddings: np.ndarray, doc_id: str) -> Dict[str, List[RetrievedPassage]]:
        """Retrieve passages for multiple claims."""
        # Transform all claims at once for efficiency
        claim_texts = [claim.claim_text for claim in claims]
        claim_embeddings = self.model.encode(claim_texts)

        # Calculate similarities for all claims
        similarities = cosine_similarity(claim_embeddings, chunk_embeddings)

        # Retrieve top-k for each claim
        results = {}
        for i, claim in enumerate(claims):
            claim_similarities = similarities[i]
            top_k_indices = np.argsort(claim_similarities)[-self.top_k:][::-1]

            retrieved = []
            for rank, idx in enumerate(top_k_indices):
                passage = RetrievedPassage(
                    passage=chunks[idx],
                    relevance_score=float(claim_similarities[idx]),
                    passage_id=f"{doc_id}_passage_{idx}"
                )
                retrieved.append(passage)

            results[claim.claim_id] = retrieved

        return results

class RetrievalAugmentedEvaluator:
    """Combine retrieval with faithfulness evaluation."""
    
    def __init__(self, retriever: PassageRetriever):
        self.retriever = retriever
        self.document_cache = {}  # Cache indexed documents
    
    def process_document(self, doc_text: str, doc_id: str) -> Tuple[List[str], np.ndarray]:
        """Process and cache a document for retrieval."""
        if doc_id not in self.document_cache:
            chunks, embeddings = self.retriever.index_document(doc_text, doc_id)
            self.document_cache[doc_id] = (chunks, embeddings)
        return self.document_cache[doc_id]
    
    def get_retrieval_context(self, claim: Claim, doc_text: str, doc_id: str) -> str:
        """Get formatted retrieval context for a claim."""
        chunks, embeddings = self.process_document(doc_text, doc_id)
        passages = self.retriever.retrieve_passages(claim, chunks, embeddings, doc_id)
        
        # Format passages for LLM consumption
        context_parts = []
        for i, passage in enumerate(passages, 1):
            context_parts.append(f"Passage {i} (relevance: {passage.relevance_score:.3f}):\n{passage.passage}")
        
        return "\n\n".join(context_parts)
