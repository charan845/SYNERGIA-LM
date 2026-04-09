"""
SYNERGIA-LM — Knowledge Graph Builder
Constructs adjacency list and index structures from triples.
"""

from collections import defaultdict
from dataset_loader import Triple, parse_relation_path


class KnowledgeGraph:
    """
    In-memory Knowledge Graph with bidirectional adjacency lists.
    Supports k-hop subgraph extraction.
    """

    def __init__(self, triples: list = None):
        # Forward: subject -> [(relation, object)]
        self.adj_forward = defaultdict(list)
        # Backward: object -> [(relation, subject)]
        self.adj_backward = defaultdict(list)
        # All triples stored
        self.triples = []
        # Index: relation -> list of (subject, object)
        self.relation_index = defaultdict(list)
        # Entity set
        self.entities = set()
        # Relation set
        self.relations = set()
        # Short relation name cache
        self.relation_short = {}

        if triples:
            self.build(triples)

    def build(self, triples: list):
        """Build graph structures from list of Triple objects."""
        for t in triples:
            self.triples.append(t)
            self.adj_forward[t.subject].append((t.relation, t.object))
            self.adj_backward[t.object].append((t.relation, t.subject))
            self.relation_index[t.relation].append((t.subject, t.object))
            self.entities.add(t.subject)
            self.entities.add(t.object)
            self.relations.add(t.relation)

            if t.relation not in self.relation_short:
                self.relation_short[t.relation] = parse_relation_path(t.relation)

    def get_neighbors(self, entity: str, direction: str = "both") -> list:
        """
        Get neighbors of an entity.
        direction: 'forward', 'backward', or 'both'
        Returns: list of (relation, neighbor_entity)
        """
        neighbors = []
        if direction in ("forward", "both"):
            neighbors.extend(self.adj_forward.get(entity, []))
        if direction in ("backward", "both"):
            neighbors.extend(self.adj_backward.get(entity, []))
        return neighbors

    def get_degree(self, entity: str) -> int:
        """Get total degree (in + out) of an entity."""
        return len(self.adj_forward.get(entity, [])) + len(self.adj_backward.get(entity, []))

    def extract_subgraph(self, seed_entities: list, max_hops: int = 2) -> dict:
        """
        Extract k-hop subgraph around seed entities.
        Returns dict with 'nodes', 'edges', 'hops_per_node'.
        """
        visited = set()
        edges = []
        hops = {}  # entity -> hop distance

        # BFS
        from collections import deque
        queue = deque()
        for e in seed_entities:
            if e in self.entities:
                queue.append((e, 0))
                visited.add(e)
                hops[e] = 0

        while queue:
            entity, dist = queue.popleft()
            if dist >= max_hops:
                continue

            neighbors = self.get_neighbors(entity)
            for rel, neighbor in neighbors:
                edges.append((entity, rel, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    hops[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return {
            "nodes": visited,
            "edges": edges,
            "hops": hops,
            "num_nodes": len(visited),
            "num_edges": len(edges),
        }

    def find_paths(self, source: str, target: str, max_length: int = 3) -> list:
        """
        Find all simple paths between source and target up to max_length edges.
        Returns list of paths, where each path is [(entity, relation, entity), ...]
        """
        if source not in self.entities or target not in self.entities:
            return []

        paths = []
        self._dfs_paths(source, target, [], set(), max_length, paths)
        return paths

    def _dfs_paths(self, current, target, path, visited, remaining, results):
        if remaining < 0:
            return
        if current == target and path:
            results.append(list(path))
            return

        visited.add(current)
        for rel, neighbor in self.get_neighbors(current):
            if neighbor not in visited:
                path.append((current, rel, neighbor))
                self._dfs_paths(neighbor, target, path, visited, remaining - 1, results)
                path.pop()

        visited.discard(current)

    def get_relation_distribution(self) -> dict:
        """Get distribution of relations in the graph."""
        dist = {}
        for rel in self.relations:
            short = self.relation_short.get(rel, rel)
            count = len(self.relation_index[rel])
            dist[short] = count
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def print_summary(self):
        """Print graph summary."""
        print(f"\n  Knowledge Graph Summary:")
        print(f"    Entities:    {len(self.entities)}")
        print(f"    Relations:   {len(self.relations)}")
        print(f"    Triples:     {len(self.triples)}")
        print(f"    Avg degree:  {2 * len(self.triples) / max(len(self.entities), 1):.2f}")

        print(f"\n    Top relations:")
        dist = self.get_relation_distribution()
        for rel, count in list(dist.items())[:10]:
            print(f"      {rel:<40} {count:>5}")

        # Degree distribution
        degrees = [self.get_degree(e) for e in self.entities]
        if degrees:
            print(f"\n    Degree statistics:")
            print(f"      Min: {min(degrees)}, Max: {max(degrees)}, "
                  f"Mean: {sum(degrees)/len(degrees):.2f}")

        # High-degree hubs
        top_hubs = sorted(self.entities, key=lambda e: self.get_degree(e), reverse=True)[:5]
        print(f"    Top hubs:")
        for e in top_hubs:
            print(f"      {e:<20} degree={self.get_degree(e)}")


if __name__ == "__main__":
    from dataset_loader import load_dataset
    data = load_dataset(".")
    kg = KnowledgeGraph(data["all"])
    kg.print_summary()

    # Test subgraph extraction
    if data["all"]:
        seed = data["all"][0].subject
        sub = kg.extract_subgraph([seed], max_hops=2)
        print(f"\n  Subgraph from {seed}: {sub['num_nodes']} nodes, {sub['num_edges']} edges")