"""Dataset loading for long-document summarization."""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from datasets import load_dataset

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
    
    def __init__(self, dataset_name: str = "sample", num_docs: int = 30, min_words: int = 2000):
        self.dataset_name = dataset_name
        self.num_docs = num_docs
        self.min_words = min_words
        self.documents: List[Document] = []
    
    def load_dataset(self) -> List[Document]:
        """Load and filter dataset."""
        if self.dataset_name == "sample":
            self._load_sample_data("sample")
        elif self.dataset_name == "govreport":
            self._load_govreport()
        elif self.dataset_name == "arxiv":
            self._load_arxiv()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        print(f"Loaded {len(self.documents)} documents from {self.dataset_name}")
        return self.documents
    
    def _load_govreport(self):
        """Load GovReport dataset."""
        print("Loading GovReport dataset from HuggingFace...")
        try:
            dataset = load_dataset("ccdv/govreport-summarization", split="test")
        except Exception as e:
            print(f"Error loading GovReport dataset: {e}")
            print("Falling back to sample data...")
            self._load_sample_data("govreport")
            return
        
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
        print("Loading arXiv dataset from HuggingFace...")
        try:
            dataset = load_dataset("scientific_papers", "arxiv", split="test")
        except Exception as e:
            print(f"Error loading arXiv dataset: {e}")
            print("Falling back to sample data...")
            self._load_sample_data("arxiv")
            return
        
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
    
    def _load_sample_data(self, dataset_type: str):
        """Load sample data when HuggingFace datasets are unavailable."""
        print(f"Loading sample {dataset_type} data for testing...")
        
        # Sample documents for testing (shorter for quick testing)
        sample_docs = {
            "sample": [
                {
                    "title": "Economic Growth Report 2023",
                    "text": "The U.S. economy showed robust growth in 2023, with GDP increasing by 2.5% in the third quarter. Unemployment rates fell to 4.1%, the lowest level in five years. Inflation remained stable at 2.0%, within the Federal Reserve's target range. Consumer spending increased by 3.2%, driven by strong labor market conditions. Business investment grew by 4.1%, indicating confidence in economic expansion. International trade showed a surplus of $2.3 billion, reflecting improved export performance. The housing market experienced moderate growth with home prices increasing by 5.2% nationally. Federal Reserve maintained interest rates at 5.25%, balancing inflation control with economic growth. Stock market performance was strong, with the S&P 500 gaining 15% for the year. Consumer confidence reached its highest level since 2000, indicating positive economic outlook." * 15,  # Reduced repetition
                    "summary": "The U.S. economy grew by 2.5% in Q3 2023, with unemployment falling to 4.1% and inflation stable at 2.0%."
                },
                {
                    "title": "Climate Policy Assessment",
                    "text": "Global carbon emissions increased by 1.2% in 2023, despite increased renewable energy adoption. The United States reduced emissions by 3.5% through clean energy investments. European Union emissions fell by 2.8% due to carbon pricing mechanisms. China's emissions grew by 4.2%, reflecting continued industrial expansion. Renewable energy accounted for 30% of global electricity generation, up from 25% in 2022. Solar capacity increased by 25% worldwide, with China leading installations. Wind energy grew by 15%, with significant additions in Europe and North America. Electric vehicle sales reached 14 million units globally, a 35% increase from 2022. Battery storage capacity doubled, addressing renewable energy intermittency challenges. Carbon capture technology received $5 billion in investment, though commercial deployment remains limited. Forest conservation efforts protected 2 million hectares of tropical rainforest. Ocean acidification continued to worsen, with pH levels dropping by 0.003 units. Extreme weather events caused $200 billion in economic damages globally." * 12,
                    "summary": "Global emissions rose 1.2% in 2023, while renewable energy reached 30% of electricity generation."
                },
                {
                    "title": "Healthcare Reform Analysis",
                    "text": "Healthcare spending reached 18% of GDP in 2023, the highest level in history. Medicare enrollment increased by 2 million beneficiaries, reflecting demographic trends. Medicaid expansion covered 5 million additional low-income individuals. Prescription drug prices increased by 6%, slower than previous years due to policy interventions. Hospital consolidation continued, with 15% of facilities merging or acquiring competitors. Telehealth usage stabilized at 15% of all medical visits, down from pandemic peaks but significantly higher than pre-pandemic levels. Mental health services saw increased demand, with 25% more patients seeking treatment. Preventive care utilization increased by 10%, indicating improved health awareness. Medical debt affected 20% of households, though average debt amounts decreased by 8%. Healthcare workforce shortages persisted, with 15% of nursing positions unfilled. Rural hospital closures continued, with 10 facilities closing in 2023. Health insurance coverage remained stable at 91% of the population." * 10,
                    "summary": "Healthcare spending reached 18% of GDP, while telehealth usage stabilized at 15% of medical visits."
                }
            ],
            "govreport": [
                {
                    "title": "Economic Growth Report 2023",
                    "text": "The U.S. economy showed robust growth in 2023, with GDP increasing by 2.5% in the third quarter. Unemployment rates fell to 4.1%, the lowest level in five years. Inflation remained stable at 2.0%, within the Federal Reserve's target range. Consumer spending increased by 3.2%, driven by strong labor market conditions. Business investment grew by 4.1%, indicating confidence in economic expansion. International trade showed a surplus of $2.3 billion, reflecting improved export performance. The housing market experienced moderate growth with home prices increasing by 5.2% nationally. Federal Reserve maintained interest rates at 5.25%, balancing inflation control with economic growth. Stock market performance was strong, with the S&P 500 gaining 15% for the year. Consumer confidence reached its highest level since 2000, indicating positive economic outlook." * 20,
                    "summary": "The U.S. economy grew by 2.5% in Q3 2023, with unemployment falling to 4.1% and inflation stable at 2.0%."
                },
                {
                    "title": "Climate Policy Assessment",
                    "text": "Global carbon emissions increased by 1.2% in 2023, despite increased renewable energy adoption. The United States reduced emissions by 3.5% through clean energy investments. European Union emissions fell by 2.8% due to carbon pricing mechanisms. China's emissions grew by 4.2%, reflecting continued industrial expansion. Renewable energy accounted for 30% of global electricity generation, up from 25% in 2022. Solar capacity increased by 25% worldwide, with China leading installations. Wind energy grew by 15%, with significant additions in Europe and North America. Electric vehicle sales reached 14 million units globally, a 35% increase from 2022. Battery storage capacity doubled, addressing renewable energy intermittency challenges. Carbon capture technology received $5 billion in investment, though commercial deployment remains limited. Forest conservation efforts protected 2 million hectares of tropical rainforest. Ocean acidification continued to worsen, with pH levels dropping by 0.003 units. Extreme weather events caused $200 billion in economic damages globally." * 15,
                    "summary": "Global emissions rose 1.2% in 2023, while renewable energy reached 30% of electricity generation."
                },
                {
                    "title": "Healthcare Reform Analysis",
                    "text": "Healthcare spending reached 18% of GDP in 2023, the highest level in history. Medicare enrollment increased by 2 million beneficiaries, reflecting demographic trends. Medicaid expansion covered 5 million additional low-income individuals. Prescription drug prices increased by 6%, slower than previous years due to policy interventions. Hospital consolidation continued, with 15% of facilities merging or acquiring competitors. Telehealth usage stabilized at 15% of all medical visits, down from pandemic peaks but significantly higher than pre-pandemic levels. Mental health services saw increased demand, with 25% more patients seeking treatment. Preventive care utilization increased by 10%, indicating improved health awareness. Medical debt affected 20% of households, though average debt amounts decreased by 8%. Healthcare workforce shortages persisted, with 15% of nursing positions unfilled. Rural hospital closures continued, with 10 facilities closing in 2023. Health insurance coverage remained stable at 91% of the population." * 12,
                    "summary": "Healthcare spending reached 18% of GDP, while telehealth usage stabilized at 15% of medical visits."
                }
            ],
            "arxiv": [
                {
                    "title": "Deep Learning for Natural Language Processing",
                    "text": "Recent advances in deep learning have revolutionized natural language processing. Transformer architectures have become the dominant approach for sequence modeling tasks. Pre-trained language models like BERT and GPT have achieved state-of-the-art results across various NLP benchmarks. Fine-tuning these models on specific tasks has become the standard paradigm. Transfer learning from large-scale pre-training has significantly reduced the need for task-specific labeled data. Multi-modal models that combine text with images have shown promising results for vision-language tasks. Efficient transformer architectures have reduced computational requirements while maintaining performance. Attention mechanisms have been extended to handle longer sequences more effectively. Self-supervised learning approaches have reduced reliance on labeled datasets. Domain adaptation techniques have improved model performance on specialized domains. Interpretability methods have been developed to understand model decisions better. Ethical considerations around bias and fairness have gained increased attention in the research community." * 18,
                    "summary": "Transformer architectures and pre-trained models have revolutionized NLP, achieving state-of-the-art results across benchmarks."
                },
                {
                    "title": "Quantum Computing Applications",
                    "text": "Quantum computing has emerged as a promising paradigm for solving computationally hard problems. Quantum algorithms for optimization have shown theoretical speedups over classical approaches. Error correction remains a significant challenge for practical quantum computing. Quantum machine learning algorithms have been developed for various data analysis tasks. Hybrid quantum-classical algorithms offer near-term practical applications. Quantum simulation has potential applications in chemistry and materials science. Quantum cryptography provides theoretically secure communication protocols. Hardware advances have increased qubit counts and reduced error rates. Cloud quantum computing services have made quantum resources more accessible. Quantum supremacy has been demonstrated for specific problem instances. Post-quantum cryptography is being developed to address future security threats. Quantum annealing has found applications in optimization problems. Quantum neural networks represent an emerging research direction." * 15,
                    "summary": "Quantum computing shows promise for optimization and simulation, though error correction remains a challenge."
                },
                {
                    "title": "Climate Modeling with Machine Learning",
                    "text": "Machine learning techniques are increasingly being applied to climate modeling. Neural networks can learn complex patterns in climate data more effectively than traditional methods. Deep learning models have improved weather prediction accuracy significantly. Ensemble methods combine multiple models for more robust predictions. Transfer learning from satellite imagery has enhanced climate monitoring capabilities. Graph neural networks model climate system interactions effectively. Uncertainty quantification methods have been developed for climate projections. Data assimilation techniques incorporate observational data into models. Extreme event prediction has improved with machine learning approaches. Long-term climate projections benefit from pattern recognition in historical data. Climate model intercomparison projects use ML for analysis. Regional climate downscaling has been enhanced with machine learning. Hybrid physics-ML models combine domain knowledge with data-driven approaches." * 14,
                    "summary": "Machine learning has improved climate modeling accuracy, particularly in weather prediction and extreme event forecasting."
                }
            ]
        }
        
        sample_data = sample_docs.get(dataset_type, sample_docs["sample"])
        
        for i, item in enumerate(sample_data[:self.num_docs]):
            doc = Document(
                doc_id=f"{dataset_type}_sample_{i}",
                title=item["title"],
                text=item["text"],
                summary=item["summary"],
                word_count=len(item["text"].split())
            )
            self.documents.append(doc)
        
        print(f"Loaded {len(self.documents)} sample documents for testing")
    
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
    # Simple sentence splitting using regex
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
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
