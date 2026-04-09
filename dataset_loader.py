"""
SYNERGIA-LM — Dataset Loader
Loads Freebase KG triples from train.txt, test.txt, valid.txt
"""

import os
from collections import defaultdict


class Triple:
    """Single Knowledge Graph triple."""
    def __init__(self, subject: str, relation: str, obj: str):
        self.subject = subject
        self.relation = relation
        self.object = obj

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object})"

    def to_tuple(self):
        return (self.subject, self.relation, self.object)


def parse_relation_path(path: str) -> str:
    """
    Extract the last two meaningful segments from a Freebase relation path.
    Example: /film/actor/film./film/performance/film -> actor_film
    """
    # Split on '/' and filter empty
    parts = [p for p in path.split("/") if p and p != "."]
    if len(parts) >= 2:
        return parts[-2] + "_" + parts[-1]
    elif len(parts) == 1:
        return parts[0]
    return path


def load_triples(filepath: str) -> list:
    """
    Load triples from a tab-separated Freebase file.
    Format: subject<TAB>relation<TAB>object
    """
    triples = []
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return triples

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if s and r and o:
                triples.append(Triple(s, r, o))

    return triples


def load_dataset(data_dir: str = ".") -> dict:
    """
    Load all three splits and compute statistics.
    Returns dict with 'train', 'test', 'valid' lists and 'stats'.
    """
    print("=" * 60)
    print("SYNERGIA-LM — Loading Freebase KG Dataset")
    print("=" * 60)

    train = load_triples(os.path.join(data_dir, "train.txt"))
    test = load_triples(os.path.join(data_dir, "test.txt"))
    valid = load_triples(os.path.join(data_dir, "valid.txt"))

    all_triples = train + test + valid

    # Compute statistics
    all_entities = set()
    all_relations = set()
    relation_counts = defaultdict(int)
    subject_counts = defaultdict(int)
    object_counts = defaultdict(int)

    for t in all_triples:
        all_entities.add(t.subject)
        all_entities.add(t.object)
        all_relations.add(t.relation)
        relation_counts[t.relation] += 1
        subject_counts[t.subject] += 1
        object_counts[t.object] += 1

    stats = {
        "total_triples": len(all_triples),
        "train_triples": len(train),
        "test_triples": len(test),
        "valid_triples": len(valid),
        "unique_entities": len(all_entities),
        "unique_relations": len(all_relations),
        "avg_triples_per_entity": round(len(all_triples) / max(len(all_entities), 1), 2),
        "top_relations": sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_entities": sorted(
            [(e, subject_counts.get(e, 0) + object_counts.get(e, 0))
             for e in all_entities],
            key=lambda x: x[1], reverse=True
        )[:15],
    }

    # Print summary
    print(f"\n  Train triples:  {stats['train_triples']}")
    print(f"  Test triples:   {stats['test_triples']}")
    print(f"  Valid triples:  {stats['valid_triples']}")
    print(f"  Total triples:  {stats['total_triples']}")
    print(f"  Unique entities: {stats['unique_entities']}")
    print(f"  Unique relations: {stats['unique_relations']}")
    print(f"  Avg triples/entity: {stats['avg_triples_per_entity']}")

    print(f"\n  Top 10 Relations:")
    for rel, count in stats["top_relations"]:
        short = parse_relation_path(rel)
        print(f"    {short:<40} {count:>5}")

    print(f"\n  Top 15 Entities (by degree):")
    for ent, deg in stats["top_entities"]:
        print(f"    {ent:<20} degree={deg}")

    return {
        "train": train,
        "test": test,
        "valid": valid,
        "all": all_triples,
        "entities": all_entities,
        "relations": all_relations,
        "stats": stats,
    }


if __name__ == "__main__":
    data = load_dataset(".")