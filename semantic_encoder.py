"""
SYNERGIA-LM — Stage 2: Semantic Encoder
Uses NLTK + TF-IDF to encode entity IDs and relations into embeddings.
Since Freebase IDs are not natural language, we encode the RELATION PATHS
(which contain meaningful English words like 'film', 'actor', etc.)
"""

import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_loader import parse_relation_path


class SemanticEncoder:
    """
    Encodes relation paths and entity IDs into vector representations.
    Uses character n-grams for entity IDs (since they're mid-format)
    and word-level features for relations.
    """

    def __init__(self):
        self._fitted = False
        self.relation_vectorizer = None
        self.entity_vectorizer = None
        self.relation_vocab = {}
        self.entity_vocab = {}

    def _entity_to_features(self, entity_id: str) -> str:
        """
        Convert Freebase entity ID to feature string.
        Uses character trigrams and segment features.
        """
        clean = entity_id.replace("/m/", "").replace("_", "")
        # Character trigrams
        trigrams = [clean[i:i+3] for i in range(len(clean) - 2)] if len(clean) > 2 else [clean]
        # Hex segments (Freebase mids encode info in hex)
        segments = [s for s in entity_id.replace("/", " ").replace("_", " ").split() if s]
        return " ".join(trigrams + segments)

    def _relation_to_features(self, relation_path: str) -> str:
        """
        Convert Freebase relation path to feature string.
        Relations contain meaningful words like 'film', 'actor', 'person'.
        """
        short = parse_relation_path(relation_path)
        # Also extract all individual words from the path
        words = relation_path.replace("/", " ").replace(".", " ").replace("_", " ").split()
        words = [w for w in words if w and len(w) > 1 and not w.startswith("m")]
        return " ".join(words + [short])

    def fit(self, entities: set, relations: set):
        """Fit vectorizers on entity and relation vocabularies."""
        entity_texts = [self._entity_to_features(e) for e in entities]
        relation_texts = [self._relation_to_features(r) for r in relations]

        self.entity_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=300,
            sublinear_tf=True,
        )
        self.entity_vectorizer.fit(entity_texts)

        self.relation_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True,
        )
        self.relation_vectorizer.fit(relation_texts)

        # Store vocab mappings
        self.entity_vocab = {e: i for i, e in enumerate(entities)}
        self.relation_vocab = {r: i for i, r in enumerate(relations)}

        self._fitted = True
        return self

    def encode_entity(self, entity_id: str) -> np.ndarray:
        """Encode a single entity ID into a vector."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted.")
        features = self._entity_to_features(entity_id)
        return self.entity_vectorizer.transform([features]).toarray()[0]

    def encode_relation(self, relation_path: str) -> np.ndarray:
        """Encode a single relation path into a vector."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted.")
        features = self._relation_to_features(relation_path)
        return self.relation_vectorizer.transform([features]).toarray()[0]

    def encode_entities_batch(self, entity_ids: list) -> dict:
        """Encode multiple entities. Returns {entity_id: vector}."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted.")
        texts = [self._entity_to_features(e) for e in entity_ids]
        vectors = self.entity_vectorizer.transform(texts).toarray()
        return {e: vectors[i] for i, e in enumerate(entity_ids)}

    def encode_relations_batch(self, relation_paths: list) -> dict:
        """Encode multiple relations. Returns {relation: vector}."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted.")
        texts = [self._relation_to_features(r) for r in relation_paths]
        vectors = self.relation_vectorizer.transform(texts).toarray()
        return {r: vectors[i] for i, r in enumerate(relation_paths)}

    def similarity_entity(self, e1: str, e2: str) -> float:
        """Cosine similarity between two entities."""
        v1 = self.encode_entity(e1).reshape(1, -1)
        v2 = self.encode_entity(e2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]

    def similarity_relation(self, r1: str, r2: str) -> float:
        """Cosine similarity between two relations."""
        v1 = self.encode_relation(r1).reshape(1, -1)
        v2 = self.encode_relation(r2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]

    def find_similar_relations(self, query_relation: str, all_relations: set, top_k: int = 5) -> list:
        """Find top-k most similar relations to a query."""
        q_vec = self.encode_relation(query_relation).reshape(1, -1)
        rel_list = list(all_relations)
        rel_vecs = self.encode_relations_batch(rel_list)
        sims = []
        for r in rel_list:
            sim = cosine_similarity(q_vec, rel_vecs[r].reshape(1, -1))[0][0]
            sims.append((r, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(r, s, parse_relation_path(r)) for r, s in sims[:top_k]]


if __name__ == "__main__":
    from dataset_loader import load_dataset
    data = load_dataset(".")

    encoder = SemanticEncoder()
    encoder.fit(data["entities"], data["relations"])

    # Test relation similarity
    print("\n  Relation Similarity Examples:")
    rels = list(data["relations"])[:5]
    for i in range(min(3, len(rels))):
        for j in range(i + 1, min(4, len(rels))):
            sim = encoder.similarity_relation(rels[i], rels[j])
            r1_short = parse_relation_path(rels[i])
            r2_short = parse_relation_path(rels[j])
            print(f"    {r1_short[:30]:<30} vs {r2_short[:30]:<30}  sim={sim:.4f}")

    # Find similar relations
    if rels:
        print(f"\n  Most similar to '{parse_relation_path(rels[0])}':")
        similar = encoder.find_similar_relations(rels[0], data["relations"], top_k=5)
        for r, s, short in similar:
            print(f"    {short:<40} {s:.4f}")