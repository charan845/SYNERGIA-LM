from semantic_encoder import SemanticEncoder
from kg_builder import KnowledgeGraph
from dataset_loader import parse_relation_path


class KGRetriever:

    def __init__(self, kg, encoder):
        self.kg = kg
        self.encoder = encoder

    def retrieve_for_completion(self, subject, relation, max_hops=2):
        subgraph = self.kg.extract_subgraph([subject], max_hops=max_hops)
        same_relation_objects = set()
        for s, o in self.kg.relation_index.get(relation, []):
            same_relation_objects.add(o)
        similar_rels = self.encoder.find_similar_relations(relation, self.kg.relations, top_k=10)
        similar_rel_objects = set()
        for rel, sim, _ in similar_rels:
            for s, o in self.kg.relation_index.get(rel, []):
                if sim > 0.3:
                    similar_rel_objects.add(o)
        candidates = same_relation_objects | similar_rel_objects | subgraph["nodes"]
        candidates.discard(subject)
        return {
            "subject": subject,
            "relation": relation,
            "relation_short": parse_relation_path(relation),
            "subgraph": subgraph,
            "same_relation_objects": same_relation_objects,
            "similar_relation_objects": similar_rel_objects,
            "candidates": candidates,
            "num_candidates": len(candidates),
        }

    def retrieve_for_evaluation(self, test_triples, max_hops=2):
        results = []
        for i, t in enumerate(test_triples):
            if i % 100 == 0 and i > 0:
                print(f"    Retrieved {i}/{len(test_triples)}...")
            result = self.retrieve_for_completion(t.subject, t.relation, max_hops)
            result["ground_truth"] = t.object
            result["in_candidates"] = t.object in result["candidates"]
            results.append(result)
        return results