import numpy as np
from collections import defaultdict
from semantic_encoder import SemanticEncoder
from kg_builder import KnowledgeGraph
from dataset_loader import parse_relation_path


class FuzzyLogicOps:

    @staticmethod
    def fuzzy_and(a, b):
        return a * b

    @staticmethod
    def fuzzy_or(a, b):
        return a + b - a * b

    @staticmethod
    def fuzzy_not(a):
        return 1.0 - a


class ReasoningLayer:

    def __init__(self, kg, encoder):
        self.kg = kg
        self.encoder = encoder
        self.logic = FuzzyLogicOps()
        self.relation_embeddings = {}
        self.entity_embeddings = {}
        self.obj_set_cache = {}

    def precompute_embeddings(self):
        print("  Precomputing embeddings...")
        ent_list = list(self.kg.entities)
        rel_list = list(self.kg.relations)
        self.entity_embeddings = self.encoder.encode_entities_batch(ent_list)
        self.relation_embeddings = self.encoder.encode_relations_batch(rel_list)
        for r in rel_list:
            objs = set()
            for s, o in self.kg.relation_index.get(r, []):
                objs.add(o)
            self.obj_set_cache[r] = objs
        print(f"    {len(self.entity_embeddings)} entities, {len(self.relation_embeddings)} relations")

    def _cosine(self, v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def score_candidate(self, subject, relation, candidate):
        rel_emb = self.relation_embeddings.get(relation)
        cand_emb = self.entity_embeddings.get(candidate)
        subj_emb = self.entity_embeddings.get(subject)

        s_direct = 0.0
        for s, o in self.kg.relation_index.get(relation, []):
            if s == subject and o == candidate:
                s_direct = 1.0
                break

        s_type = 0.0
        if cand_emb is not None:
            objs = self.obj_set_cache.get(relation, set())
            obj_list = list(objs)[:30]
            sims = []
            for obj in obj_list:
                obj_emb = self.entity_embeddings.get(obj)
                if obj_emb is not None:
                    sims.append(self._cosine(cand_emb, obj_emb))
            if sims:
                s_type = np.mean(sims)

        s_onehop = 0.0
        neighbors = self.kg.get_neighbors(subject)
        count = 0
        for rel, neighbor in neighbors:
            if count >= 20:
                break
            targets = self.kg.get_neighbors(neighbor, direction="forward")
            for rel2, target in targets:
                if target == candidate:
                    sim1 = self.encoder.similarity_relation(relation, rel)
                    sim2 = self.encoder.similarity_relation(relation, rel2)
                    path = self.logic.fuzzy_and(sim1, sim2)
                    s_onehop = self.logic.fuzzy_or(s_onehop, path)
            count += 1

        s_combined = self.logic.fuzzy_and(s_onehop, s_type)
        s_final = self.logic.fuzzy_or(s_direct, s_combined * 0.8)

        return {
            "candidate": candidate,
            "score_direct": round(s_direct, 4),
            "score_onehop": round(s_onehop, 4),
            "score_type": round(s_type, 4),
            "score_final": round(s_final, 4),
        }

    def rank_candidates(self, subject, relation, candidates, top_k=10):
        cand_list = list(candidates)[:100]
        scores = []
        for cand in cand_list:
            result = self.score_candidate(subject, relation, cand)
            scores.append(result)
        scores.sort(key=lambda x: x["score_final"], reverse=True)
        return scores[:top_k]