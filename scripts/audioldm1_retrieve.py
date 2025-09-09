"""
CLAP-AudioLDM Retrieval Pipeline
Complete implementation for audio retrieval and generation using CLAP embeddings
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import pickle
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import librosa
import soundfile as sf

# For AudioLDM - you may need to install: pip install audioldm
try:
    from audioldm import build_model, text_to_audio
    from audioldm.audio.stft import TacotronSTFT
    from audioldm.variational_autoencoder import AutoencoderKL
except ImportError:
    print("Please install AudioLDM: pip install audioldm")

# Configuration
@dataclass
class Config:
    """Configuration for the retrieval pipeline"""
    embedding_dim: int = 512
    batch_size: int = 32
    top_k: int = 10
    similarity_metric: str = 'cosine'  # 'cosine', 'euclidean', 'dot'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir: str = './cache'
    output_dir: str = './outputs'
    audioldm_model: str = 'audioldm-s-full'
    sample_rate: int = 16000
    audio_length: float = 10.0  # seconds
    guidance_scale: float = 2.5
    ddim_steps: int = 100
    n_candidates: int = 1


class EmbeddingDataset(Dataset):
    """Dataset class for handling embeddings"""
    
    def __init__(self, embeddings: np.ndarray, metadata: Optional[Dict] = None):
        self.embeddings = embeddings
        self.metadata = metadata or {}
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            'index': idx,
            'metadata': self.metadata.get(idx, {})
        }


class CLAPRetrievalSystem:
    """Main retrieval system using CLAP embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize AudioLDM
        print("Loading AudioLDM model...")
        self.audioldm = self._load_audioldm()
        
        # Storage for embeddings
        self.query_embeddings = {}
        self.retrieval_embeddings = None
        self.similarity_cache = {}
        
    def _load_audioldm(self):
        """Load AudioLDM model"""
        try:
            model = build_model(model_name=self.config.audioldm_model)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Warning: Could not load AudioLDM model: {e}")
            return None
    
    def load_embeddings(self, embedding_path: str) -> Dict[str, np.ndarray]:
        """Load pre-computed CLAP embeddings"""
        print(f"Loading embeddings from {embedding_path}")
        
        # Load the numpy file
        data = np.load(embedding_path, allow_pickle=True)
        
        # Handle different possible formats
        if isinstance(data, np.ndarray):
            # If it's a plain array, we need to know the structure
            return self._parse_embedding_array(data)
        elif hasattr(data, 'item'):
            # If it's a numpy object array containing a dict
            return data.item()
        else:
            # If it's already a dict-like object
            return dict(data)
    
    def _parse_embedding_array(self, arr: np.ndarray) -> Dict[str, np.ndarray]:
        """Parse embedding array based on expected structure"""
        # This assumes a specific structure - adjust based on your actual file
        # Example: first part is MAESTRO, second is URMP, rest is FSD50K
        
        print(f"Embedding array shape: {arr.shape}")
        
        # You'll need to adjust these indices based on your actual data
        result = {}
        
        # Example partitioning (adjust based on your data):
        # Assuming you know the sizes of each dataset
        maestro_size = 1000  # Replace with actual size
        urmp_size = 500      # Replace with actual size
        
        idx = 0
        if arr.ndim == 2:
            result['maestro'] = arr[idx:idx+maestro_size]
            idx += maestro_size
            
            result['urmp'] = arr[idx:idx+urmp_size]
            idx += urmp_size
            
            result['fsd50k'] = arr[idx:]
        else:
            # Handle 3D array or other formats
            print("Warning: Unexpected embedding array format")
            result['all'] = arr
            
        return result
    
    def setup_retrieval_corpus(self, corpus_embeddings: np.ndarray, 
                              corpus_metadata: Optional[Dict] = None):
        """Set up the retrieval corpus (FSD50K)"""
        print(f"Setting up retrieval corpus with {len(corpus_embeddings)} items")
        
        # Normalize embeddings for cosine similarity
        if self.config.similarity_metric == 'cosine':
            corpus_embeddings = normalize(corpus_embeddings, axis=1)
        
        self.retrieval_embeddings = corpus_embeddings
        self.retrieval_metadata = corpus_metadata or {}
        
        # Build index for fast retrieval (optional: use FAISS for large-scale)
        if len(corpus_embeddings) > 10000:
            self._build_faiss_index(corpus_embeddings)
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        try:
            import faiss
            
            d = embeddings.shape[1]
            
            if self.config.similarity_metric == 'cosine':
                # Use Inner Product for normalized vectors (cosine similarity)
                self.index = faiss.IndexFlatIP(d)
            else:
                # Use L2 distance
                self.index = faiss.IndexFlatL2(d)
            
            self.index.add(embeddings.astype(np.float32))
            print(f"Built FAISS index with {self.index.ntotal} vectors")
        except ImportError:
            print("FAISS not installed. Using numpy for similarity search.")
            self.index = None
    
    def retrieve(self, query_embedding: np.ndarray, 
                 top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k similar items from corpus
        
        Returns:
            indices: Top-k indices in the retrieval corpus
            scores: Similarity scores
        """
        top_k = top_k or self.config.top_k
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        if self.config.similarity_metric == 'cosine':
            query_embedding = normalize(query_embedding, axis=1)
        
        # Use FAISS if available
        if hasattr(self, 'index') and self.index is not None:
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), top_k
            )
            return indices[0], scores[0]
        
        # Fallback to numpy
        return self._numpy_retrieve(query_embedding, top_k)
    
    def _numpy_retrieve(self, query_embedding: np.ndarray, 
                       top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numpy-based retrieval"""
        
        if self.config.similarity_metric == 'cosine':
            # Cosine similarity
            similarities = np.dot(self.retrieval_embeddings, query_embedding.T).squeeze()
        elif self.config.similarity_metric == 'dot':
            # Dot product
            similarities = np.dot(self.retrieval_embeddings, query_embedding.T).squeeze()
        else:
            # Euclidean distance (convert to similarity)
            distances = cdist(query_embedding, self.retrieval_embeddings, 
                            metric='euclidean').squeeze()
            similarities = -distances
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def batch_retrieve(self, query_embeddings: np.ndarray, 
                      top_k: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Batch retrieval for multiple queries"""
        results = []
        
        for query_emb in tqdm(query_embeddings, desc="Retrieving"):
            indices, scores = self.retrieve(query_emb, top_k)
            results.append((indices, scores))
        
        return results
    
    def generate_audio_from_embedding(self, embedding: np.ndarray,
                                     prompt: Optional[str] = None) -> np.ndarray:
        """Generate audio using AudioLDM conditioned on CLAP embedding"""
        
        if self.audioldm is None:
            print("AudioLDM model not loaded")
            return np.zeros((self.config.sample_rate * int(self.config.audio_length),))
        
        # Convert to torch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
        
        if embedding_tensor.ndim == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        
        # Generate audio
        with torch.no_grad():
            # AudioLDM generation (adjust based on actual API)
            waveform = self.audioldm.generate_audio_from_embedding(
                embedding=embedding_tensor,
                prompt=prompt,
                guidance_scale=self.config.guidance_scale,
                ddim_steps=self.config.ddim_steps,
                n_candidate_gen_per_text=self.config.n_candidates,
                duration=self.config.audio_length
            )
        
        return waveform.cpu().numpy()
    
    def retrieval_augmented_generation(self, query_embedding: np.ndarray,
                                      blend_retrieved: bool = True,
                                      alpha: float = 0.7) -> np.ndarray:
        """
        Generate audio using retrieval-augmented generation
        
        Args:
            query_embedding: Query CLAP embedding
            blend_retrieved: Whether to blend multiple retrieved embeddings
            alpha: Weight for query vs retrieved embeddings
        """
        
        # Retrieve similar embeddings
        indices, scores = self.retrieve(query_embedding, top_k=3)
        
        # Get retrieved embeddings
        retrieved_embeddings = self.retrieval_embeddings[indices]
        
        if blend_retrieved:
            # Weighted average of retrieved embeddings based on scores
            weights = F.softmax(torch.tensor(scores), dim=0).numpy()
            blended_retrieved = np.average(retrieved_embeddings, weights=weights, axis=0)
            
            # Blend with query embedding
            final_embedding = alpha * query_embedding + (1 - alpha) * blended_retrieved
        else:
            # Use top retrieved embedding
            final_embedding = retrieved_embeddings[0]
        
        # Generate audio
        audio = self.generate_audio_from_embedding(final_embedding)
        
        return audio, indices, scores


class EvaluationMetrics:
    """Evaluation metrics for retrieval"""
    
    @staticmethod
    def mean_average_precision(retrieved: List[int], 
                              relevant: List[int]) -> float:
        """Calculate Mean Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        ap = 0.0
        relevant_found = 0
        
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                relevant_found += 1
                ap += relevant_found / (i + 1)
        
        return ap / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[int], 
                           relevant: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant)
        
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[int], 
                   relevant: List[int], 
                   k: int) -> float:
        """Calculate Recall@K"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_k = set(retrieved[:k])
        
        return len(relevant_set & retrieved_k) / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(retrieved: List[int], 
                 relevant: List[int], 
                 k: int) -> float:
        """Calculate NDCG@K"""
        def dcg(scores):
            return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))
        
        relevant_set = set(relevant)
        scores = [1 if item in relevant_set else 0 for item in retrieved[:k]]
        
        if sum(scores) == 0:
            return 0.0
        
        actual_dcg = dcg(scores)
        ideal_scores = sorted(scores, reverse=True)
        ideal_dcg = dcg(ideal_scores)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class RetrievalPipeline:
    """Complete retrieval and evaluation pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.retrieval_system = CLAPRetrievalSystem(config)
        self.metrics = EvaluationMetrics()
        
    def run_complete_pipeline(self, embedding_path: str,
                             output_results: bool = True):
        """Run the complete retrieval pipeline"""
        
        # Load embeddings
        embeddings = self.retrieval_system.load_embeddings(embedding_path)
        
        # Setup datasets
        print("\nSetting up datasets...")
        maestro_embeds = embeddings.get('maestro', np.array([]))
        urmp_embeds = embeddings.get('urmp', np.array([]))
        fsd50k_embeds = embeddings.get('fsd50k', np.array([]))
        
        print(f"MAESTRO: {len(maestro_embeds)} embeddings")
        print(f"URMP: {len(urmp_embeds)} embeddings")
        print(f"FSD50K: {len(fsd50k_embeds)} embeddings")
        
        # Setup retrieval corpus
        self.retrieval_system.setup_retrieval_corpus(fsd50k_embeds)
        
        # Combine query embeddings
        query_embeddings = np.vstack([maestro_embeds, urmp_embeds])
        query_labels = (['maestro'] * len(maestro_embeds) + 
                       ['urmp'] * len(urmp_embeds))
        
        # Run retrieval
        print("\nRunning retrieval...")
        results = self.run_retrieval_experiments(query_embeddings, query_labels)
        
        # Evaluate
        print("\nEvaluating results...")
        evaluation_results = self.evaluate_retrieval(results)
        
        # Generate audio samples
        print("\nGenerating audio samples...")
        self.generate_sample_outputs(query_embeddings[:5], results[:5])
        
        if output_results:
            self.save_results(results, evaluation_results)
        
        return results, evaluation_results
    
    def run_retrieval_experiments(self, query_embeddings: np.ndarray,
                                 query_labels: List[str]) -> List[Dict]:
        """Run retrieval experiments"""
        results = []
        
        for i, (query_emb, label) in enumerate(tqdm(
            zip(query_embeddings, query_labels), 
            total=len(query_embeddings),
            desc="Retrieving"
        )):
            indices, scores = self.retrieval_system.retrieve(query_emb)
            
            results.append({
                'query_idx': i,
                'query_label': label,
                'retrieved_indices': indices,
                'scores': scores
            })
        
        return results
    
    def evaluate_retrieval(self, results: List[Dict]) -> Dict:
        """Evaluate retrieval results"""
        
        # For demonstration - you would need actual ground truth
        # This creates synthetic ground truth for testing
        eval_metrics = {
            'map': [],
            'mrr': [],
            'recall@5': [],
            'recall@10': [],
            'ndcg@10': []
        }
        
        for result in results:
            # Synthetic ground truth (replace with actual)
            # In practice, you'd have annotations of relevant items
            retrieved = result['retrieved_indices'].tolist()
            
            # Create synthetic relevant items for demonstration
            # Replace this with actual ground truth
            relevant = list(range(5))  # First 5 items are "relevant"
            
            eval_metrics['map'].append(
                self.metrics.mean_average_precision(retrieved, relevant)
            )
            eval_metrics['mrr'].append(
                self.metrics.mean_reciprocal_rank(retrieved, relevant)
            )
            eval_metrics['recall@5'].append(
                self.metrics.recall_at_k(retrieved, relevant, 5)
            )
            eval_metrics['recall@10'].append(
                self.metrics.recall_at_k(retrieved, relevant, 10)
            )
            eval_metrics['ndcg@10'].append(
                self.metrics.ndcg_at_k(retrieved, relevant, 10)
            )
        
        # Compute averages
        avg_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in avg_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return avg_metrics
    
    def generate_sample_outputs(self, query_embeddings: np.ndarray,
                               retrieval_results: List[Dict]):
        """Generate sample audio outputs"""
        
        print("\nGenerating audio samples...")
        
        for i, (query_emb, result) in enumerate(zip(query_embeddings, retrieval_results)):
            print(f"Generating sample {i+1}/{len(query_embeddings)}")
            
            # Retrieval-augmented generation
            audio, indices, scores = self.retrieval_system.retrieval_augmented_generation(
                query_emb, blend_retrieved=True
            )
            
            # Save audio
            output_path = os.path.join(
                self.config.output_dir, 
                f"generated_sample_{i}_{result['query_label']}.wav"
            )
            
            sf.write(output_path, audio, self.config.sample_rate)
            print(f"Saved: {output_path}")
    
    def save_results(self, retrieval_results: List[Dict], 
                    evaluation_metrics: Dict):
        """Save results to disk"""
        
        # Save retrieval results
        results_path = os.path.join(self.config.output_dir, 'retrieval_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(retrieval_results, f)
        
        # Save evaluation metrics
        metrics_path = os.path.join(self.config.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2)
        
        print(f"\nResults saved to {self.config.output_dir}")
    
    def analyze_cross_dataset_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance across different query datasets"""
        
        maestro_results = [r for r in results if r['query_label'] == 'maestro']
        urmp_results = [r for r in results if r['query_label'] == 'urmp']
        
        analysis = {
            'maestro': {
                'count': len(maestro_results),
                'avg_top_score': np.mean([r['scores'][0] for r in maestro_results]),
                'avg_top5_score': np.mean([np.mean(r['scores'][:5]) for r in maestro_results])
            },
            'urmp': {
                'count': len(urmp_results),
                'avg_top_score': np.mean([r['scores'][0] for r in urmp_results]),
                'avg_top5_score': np.mean([np.mean(r['scores'][:5]) for r in urmp_results])
            }
        }
        
        print("\n" + "="*50)
        print("CROSS-DATASET ANALYSIS")
        print("="*50)
        for dataset, metrics in analysis.items():
            print(f"\n{dataset.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        return analysis


def main():
    """Main execution function"""
    
    # Configuration
    config = Config(
        embedding_dim=512,
        batch_size=32,
        top_k=10,
        similarity_metric='cosine',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir='./cache',
        output_dir='./outputs',
        audioldm_model='audioldm-s-full',
        sample_rate=16000,
        audio_length=10.0,
        guidance_scale=2.5,
        ddim_steps=100,
        n_candidates=1
    )
    
    # Initialize pipeline
    pipeline = RetrievalPipeline(config)
    
    # Run complete pipeline
    embedding_path = 'arg.npy'  # Your embedding file
    
    try:
        results, evaluation = pipeline.run_complete_pipeline(
            embedding_path=embedding_path,
            output_results=True
        )
        
        # Additional analysis
        cross_dataset_analysis = pipeline.analyze_cross_dataset_performance(results)
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        
        # Print summary
        print(f"\nProcessed {len(results)} queries")
        print(f"Generated audio samples in: {config.output_dir}")
        print(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()