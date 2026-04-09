from reasoning_layer import FuzzyLogicOps


class ConstraintEngine:

    def __init__(self):
        self.logic = FuzzyLogicOps()

    def initialize_domains(self, candidates, base_score=0.5):
        domain = {}
        for c in candidates:
            domain[c] = base_score
        return domain

    def apply_functional_constraint(self, domain, subject, relation, kg_adj):
        existing_objects = set()
        for rel, obj in kg_adj.get(subject, []):
            if rel == relation:
                existing_objects.add(obj)
        if not existing_objects:
            return domain
        new_domain = {}
        for cand, prob in domain.items():
            if cand in existing_objects:
                new_domain[cand] = self.logic.fuzzy_or(prob, 0.9)
            else:
                new_domain[cand] = prob * 0.95
        total = sum(new_domain.values())
        if total > 0:
            for cand in new_domain:
                new_domain[cand] /= total
        return new_domain

    def apply_type_constraint(self, domain, relation, type_entities, strength=0.8):
        new_domain = {}
        for cand, prob in domain.items():
            if cand in type_entities:
                new_domain[cand] = self.logic.fuzzy_or(prob, strength)
            else:
                new_domain[cand] = prob * (1.0 - strength * 0.3)
        total = sum(new_domain.values())
        if total > 0:
            for cand in new_domain:
                new_domain[cand] /= total
        return new_domain

    def apply_uniqueness_constraint(self, domain, excluded=None):
        if excluded is None:
            excluded = set()
        new_domain = {}
        for cand, prob in domain.items():
            if cand in excluded:
                new_domain[cand] = prob * 0.01
            else:
                new_domain[cand] = prob
        total = sum(new_domain.values())
        if total > 0:
            for cand in new_domain:
                new_domain[cand] /= total
        return new_domain

    def arc_consistency_pass(self, domain, constraints):
        current = dict(domain)
        for constraint in constraints:
            ctype = constraint[0]
            if ctype == "functional":
                _, subject, relation, kg_adj = constraint
                current = self.apply_functional_constraint(current, subject, relation, kg_adj)
            elif ctype == "type":
                _, relation, type_entities, strength = constraint
                current = self.apply_type_constraint(current, relation, type_entities, strength)
            elif ctype == "uniqueness":
                _, excluded = constraint[1], constraint[2] if len(constraint) > 2 else set()
                current = self.apply_uniqueness_constraint(current, excluded)
        return current

    def propagate(self, domain, constraints, max_iterations=5, convergence_threshold=0.001):
        current = dict(domain)
        for iteration in range(max_iterations):
            new_domain = self.arc_consistency_pass(current, constraints)
            max_change = 0
            for cand in current:
                change = abs(new_domain.get(cand, 0) - current.get(cand, 0))
                max_change = max(max_change, change)
            current = new_domain
            if max_change < convergence_threshold:
                break
        return current

    def merge_with_reasoning_scores(self, domain, reasoning_scores, reasoning_weight=0.7):
        merged = {}
        for cand in domain:
            c_score = domain.get(cand, 0)
            r_score = reasoning_scores.get(cand, 0)
            if c_score > 0 and r_score > 0:
                merged[cand] = (c_score ** reasoning_weight) * (r_score ** (1 - reasoning_weight))
            elif r_score > 0:
                merged[cand] = r_score
            else:
                merged[cand] = c_score
        total = sum(merged.values())
        if total > 0:
            for cand in merged:
                merged[cand] /= total
        return merged

    def get_top_predictions(self, domain, top_k=10):
        ranked = sorted(domain.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]