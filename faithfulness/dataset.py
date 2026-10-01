"""Dataset loading for long-document summarization."""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import nltk
from datasets import load_dataset

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class Document:
    """Represents a long document for summarization."""
    doc_id: str
    title: str
    text: str
    summary: Optional[str] = None  # Reference summary if available
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())

class DatasetLoader:
    """Load and process long-document summarization datasets."""
    
    def __init__(self, dataset_name: str = "govreport", num_docs: int = 30, min_words: int = 2000):
        self.dataset_name = dataset_name
        self.num_docs = num_docs
        self.min_words = min_words
        self.documents: List[Document] = []
    
    def load_dataset(self) -> List[Document]:
        """Load and filter dataset."""
        if self.dataset_name == "govreport":
            self._load_govreport()
        elif self.dataset_name == "arxiv":
            self._load_arxiv()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        print(f"Loaded {len(self.documents)} documents from {self.dataset_name}")
        return self.documents
    
    def _load_govreport(self):
        """Load GovReport dataset."""
        print("Loading GovReport dataset...")
        dataset = load_dataset("ccdv/govreport-summarization", split="test")
        
        # Filter by word count and limit
        count = 0
        for item in dataset:
            if count >= self.num_docs:
                break
            
            text = item["report"]
            word_count = len(text.split())
            
            if word_count >= self.min_words:
                doc = Document(
                    doc_id=f"govreport_{count}",
                    title=item.get("title", f"Document {count}"),
                    text=text,
                    summary=item.get("summary"),
                    word_count=word_count
                )
                self.documents.append(doc)
                count += 1
        
        print(f"Filtered to {len(self.documents)} documents with {self.min_words}+ words")
    
    def _load_arxiv(self):
        """Load arXiv dataset (scientific papers)."""
        print("Loading arXiv dataset...")
        # Use scientific_papers dataset from Hugging Face
        dataset = load_dataset("scientific_papers", "arxiv", split="test")
        
        # Filter by word count and limit
        count = 0
        for item in dataset:
            if count >= self.num_docs:
                break
            
            text = item["article"]
            word_count = len(text.split())
            
            if word_count >= self.min_words:
                doc = Document(
                    doc_id=f"arxiv_{count}",
                    title=item.get("article_name", f"Paper {count}"),
                    text=text,
                    summary=item.get("abstract"),
                    word_count=word_count
                )
                self.documents.append(doc)
                count += 1
        
        print(f"Filtered to {len(self.documents)} documents with {self.min_words}+ words")
    
    def save_documents(self, output_path: str):
        """Save documents to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        docs_data = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "summary": doc.summary,
                "word_count": doc.word_count
            }
            for doc in self.documents
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"Saved {len(self.documents)} documents to {output_path}")
    
    def load_documents_from_json(self, input_path: str) -> List[Document]:
        """Load documents from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        self.documents = [
            Document(
                doc_id=doc["doc_id"],
                title=doc["title"],
                text=doc["text"],
                summary=doc.get("summary"),
                word_count=doc["word_count"]
            )
            for doc in docs_data
        ]
        
        print(f"Loaded {len(self.documents)} documents from {input_path}")
        return self.documents

def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split document into overlapping chunks for retrieval."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_words + sentence_words > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap sentences
            overlap_sentences = []
            overlap_words = 0
            for sent in reversed(current_chunk):
                sent_words = len(sent.split())
                if overlap_words + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                else:
                    break
            current_chunk = overlap_sentences
            current_words = overlap_words
        
        current_chunk.append(sentence)
        current_words += sentence_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
