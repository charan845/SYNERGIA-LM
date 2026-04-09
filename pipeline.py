"""
SYNERGIA-LM — Full Pipeline
Stages 1-7: Input → Encode → Extract → Retrieve → Reason → Constrain → Decode
"""

import time
import numpy as np
from dataset_loader import load_dataset, parse_relation_path
from kg_builder import KnowledgeGraph
from semantic_encoder import SemanticEncoder
from kg_retriever import KGRetriever
from reasoning_layer import ReasoningLayer
from constraint_engine import ConstraintEngine
from evaluation import evaluate_predictions, print_evaluation_results


class SYNERGIALM:
    """
    Complete SYNERGIA-LM pipeline for Knowledge Graph Completion.
    Given (subject, relation, ?), predict the missing object.
    """

    def __init__(self):
        self.kg = None
        self.encoder = None
        self.retriever = None
        self.reasoner = None
        self.constraints = None
        self.data = None

    def build(self, data_dir: str = ".", max_hops: int = 2):
        """Build all pipeline components from data."""
        print("\n" + "=" * 60)
        print("  SYNERGIA-LM — Building Pipeline")
        print("=" * 60)

        t0 = time.time()

        # Stage 1: Load dataset
        print("\n  [Stage 1] Loading dataset...")
        self.data = load_dataset(data_dir)

        # Stage 2: Build KG from training data
        print("\n  [Stage 2] Building Knowledge Graph...")
        self.kg = KnowledgeGraph(self.data["train"])
        self.kg.print_summary()

        # Stage 3: Semantic Encoder
        print("\n  [Stage 3] Training Semantic Encoder...")
        self.encoder = SemanticEncoder()
        self.encoder.fit(self.kg.entities, self.kg.relations)

        # Stage 4: KG Retriever
        print("\n  [Stage 4] Initializing KG Retriever...")
        self.retriever = KGRetriever(self.kg, self.encoder)

        # Stage 5: Reasoning Layer
        print("\n  [Stage 5] Initializing Reasoning Layer...")
        self.reasoner = ReasoningLayer(self.kg, self.encoder)
        self.reasoner.precompute_embeddings()

        # Stage 6: Constraint Engine
        print("\n  [Stage 6] Initializing Constraint Engine...")
        self.constraints = ConstraintEngine()

        t1 = time.time()
        print(f"\n  Pipeline built in {t1 - t0:.2f}s")
        self.max_hops = max_hops
        return self

    def predict_single(self, subject: str, relation: str,
                       top_k: int = 10, verbose: bool = False) -> list:
        """
        Run full pipeline on a single (subject, relation, ?) query.
        Returns ranked list of [(candidate, score), ...]
        """
        rel_short = parse_relation_path(relation)

        # Stage 4: KG Retrieval
        retrieval = self.retriever.retrieve_for_completion(
            subject, relation, max_hops=self.max_hops
        )
        candidates = retrieval["candidates"]

        if not candidates:
            return []

        # Stage 5: Differentiable Symbolic Reasoning
        rankings = self.reasoner.rank_candidates(
            subject, relation, candidates, top_k=min(len(candidates), 50)
        )

        # Build reasoning score dict
        reasoning_scores = {r["candidate"]: r["score_final"] for r in rankings}

        # Stage 6: Constraint Propagation
        # Initialize soft domain
        domain = self.constraints.initialize_domains(candidates, base_score=0.1)

        # Apply constraints
        kg_adj_forward = dict(self.kg.adj_forward)
        type_entities = retrieval["same_relation_objects"]

        constraint_list = [
            ("functional", subject, relation, kg_adj_forward),
            ("type", relation, type_entities, 0.7),
        ]

        propagated = self.constraints.propagate(
            domain, constraint_list, max_iterations=3
        )

        # Merge reasoning + constraint scores
        merged = self.constraints.merge_with_reasoning_scores(
            propagated, reasoning_scores, reasoning_weight=0.6
        )

        # Stage 7: Decode — sort and return top-k
        final_ranking = self.constraints.get_top_predictions(merged, top_k)

        if verbose:
            print(f"\n  Query: ({subject}, {rel_short}, ?)")
            print(f"  Candidates: {len(candidates)}")
            print(f"  Top-{top_k} predictions:")
            for cand, score in final_ranking:
                print(f"    {cand:<25} score={score:.4f}")

        return final_ranking

    def evaluate(self, num_test: int = None, top_k: int = 10) -> dict:
        """
        Evaluate on test set. Returns metrics dict.
        """
        test_triples = self.data["test"]
        if num_test:
            test_triples = test_triples[:num_test]

        print(f"\n  Evaluating on {len(test_triples)} test triples...")
        all_predictions = []
        all_gt = []

        t0 = time.time()
        for i, t in enumerate(test_triples):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    Processing {i + 1}/{len(test_triples)}...")

            ranking = self.predict_single(t.subject, t.relation, top_k=top_k)
            all_predictions.append(ranking)
            all_gt.append(t.object)

        t1 = time.time()
        print(f"    Evaluation completed in {t1 - t0:.2f}s")

        results = evaluate_predictions(all_predictions, all_gt, ks=[1, 3, 10])
        results["eval_time"] = t1 - t0
        results["num_test"] = len(test_triples)
        return results

    def print_architecture(self):
        """Print the SYNERGIA-LM architecture diagram."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    SYNERGIA-LM Architecture                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────────────┐                                     ║
║  │  Input: (s, r, ?)   │  Stage 1: Query Input              ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  Semantic Encoder    │  Stage 2: TF-IDF + Char n-grams   ║
║  │  (NLTK + sklearn)   │  Entity & Relation Embeddings      ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  Entity/Relation    │  Stage 3: ID-based Extraction       ║
║  │  Extractor          │  Query Decomposition               ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  KG Retrieval       │  Stage 4: k-hop Subgraph           ║
║  │  Module             │  Candidate Generation              ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  ⭐ Differentiable  │  Stage 5: Fuzzy Logic              ║
║  │  Symbolic Reasoning │  AND, OR, NOT, IMPLIES            ║
║  │  Layer              │  Multi-hop Path Scoring            ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  ⭐ Constraint      │  Stage 6: Soft Arc Consistency     ║
║  │  Propagation Engine │  Domain Reduction + Pruning       ║
║  └──────────┬──────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  Score Merge &      │  Stage 7: Weighted Combination     ║
║  │  Decode             │  Final Ranking Output              ║
║  └─────────────────────┘                                     ║
║             ▼                                                ║
║  ┌─────────────────────┐                                     ║
║  │  Reliable Output    │  Top-K Predicted Objects           ║
║  │  (Ranked Candidates)│  with Confidence Scores            ║
║  └─────────────────────┘                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    pipeline = SYNERGIALM()
    pipeline.print_architecture()
    pipeline.build(data_dir=".", max_hops=2)

    # Quick demo on 3 test examples
    print("\n" + "=" * 60)
    print("  Quick Demo — 3 Test Examples")
    print("=" * 60)
    for t in pipeline.data["test"][:3]:
        pipeline.predict_single(t.subject, t.relation, top_k=5, verbose=True)
        print(f"  Ground Truth: {t.object}")
        print()

    # Full evaluation
    results = pipeline.evaluate(num_test=200, top_k=10)
    print_evaluation_results(results, "SYNERGIA-LM (Baseline)")