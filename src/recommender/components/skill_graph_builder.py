import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from src.recommender.entity.config_entity import SkillGraphConfig
from src.recommender.entity.artifact_entity import SkillGraphArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class SkillGraphBuilder:
    """Loads skill_prerequisites into a directed acyclic graph (edge =
    prerequisite -> dependent, so topological order is a valid learning
    order) and persists it for the recommender to traverse at serving
    time - graph operations, not recursive SQL, is what path generation
    actually needs."""

    def __init__(self, config: SkillGraphConfig) -> None:
        self.config = config

    def initiate_graph_build(self) -> SkillGraphArtifact:
        try:
            logging.info("Building skill prerequisite graph")
            skills = pd.read_csv(self.config.skills_path)
            edges = pd.read_csv(self.config.prerequisite_edges_path)

            graph = nx.DiGraph()
            for skill_id in skills["skill_id"]:
                graph.add_node(skill_id)
            for _, row in edges.iterrows():
                graph.add_edge(row["prerequisite_skill_id"], row["skill_id"])

            if not nx.is_directed_acyclic_graph(graph):
                cycle = nx.find_cycle(graph)
                raise ValueError(f"Skill prerequisite graph has a cycle: {cycle}")

            Path(self.config.graph_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.graph_cache_path, "wb") as f:
                pickle.dump(graph, f)

            logging.info(
                f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            return SkillGraphArtifact(
                graph_object_path=self.config.graph_cache_path,
                num_nodes=graph.number_of_nodes(),
                num_edges=graph.number_of_edges(),
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e


def get_missing_skills_ordered(
    graph: nx.DiGraph,
    possessed_skills: set | None,
    required_skills: set,
    mastery_states: dict | None = None,
) -> list:
    """Return required missing skills in prerequisite-safe order.

    A validated skill is a *hard boundary*: it is considered mastered and is
    never reintroduced merely because a downstream goal depends on it.
    Skills marked ``needs_review`` or ``failed`` are *not* boundaries, so they
    are included again before downstream dependents. ``possessed_skills`` is
    retained as a backward-compatible fallback for callers that predate the
    explicit mastery-state contract.
    """
    possessed = set(possessed_skills or set())
    if mastery_states is not None:
        validated = set()
        for skill_id, raw in mastery_states.items():
            status = (
                raw.get("status")
                if isinstance(raw, dict)
                else getattr(raw, "status", None)
            )
            if status == "validated":
                validated.add(skill_id)
        possessed = validated

    missing = {s for s in required_skills if s not in possessed}
    closure = set(missing)
    for skill in missing:
        if skill in graph:
            closure |= {a for a in nx.ancestors(graph, skill) if a not in possessed}

    if not closure:
        return []

    subgraph = graph.subgraph(closure)
    depth = {n: 0 for n in subgraph.nodes if subgraph.in_degree(n) == 0}
    for node in nx.topological_sort(subgraph):
        preds = list(subgraph.predecessors(node))
        if preds:
            depth[node] = 1 + max(depth[p] for p in preds)
        else:
            depth.setdefault(node, 0)

    return sorted(subgraph.nodes, key=lambda n: (depth[n], n))
