import unittest

import networkx as nx

from src.recommender.components.skill_graph_builder import get_missing_skills_ordered


class TestSkillGraphTraversal(unittest.TestCase):
    def setUp(self):
        # python -> {statistics, sql} ; statistics -> ml ; sql -> data_modeling
        self.graph = nx.DiGraph()
        edges = [
            ("python", "statistics"),
            ("python", "sql"),
            ("statistics", "ml"),
            ("sql", "data_modeling"),
        ]
        self.graph.add_edges_from(edges)

    def test_cycle_is_detected(self):
        cyclic = nx.DiGraph()
        cyclic.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        self.assertFalse(nx.is_directed_acyclic_graph(cyclic))

    def test_prerequisites_always_precede_dependents(self):
        ordered = get_missing_skills_ordered(
            self.graph, possessed_skills=set(), required_skills={"ml", "data_modeling"}
        )
        self.assertEqual(set(ordered), {"python", "statistics", "sql", "ml", "data_modeling"})
        self.assertLess(ordered.index("python"), ordered.index("statistics"))
        self.assertLess(ordered.index("python"), ordered.index("sql"))
        self.assertLess(ordered.index("statistics"), ordered.index("ml"))
        self.assertLess(ordered.index("sql"), ordered.index("data_modeling"))

    def test_already_possessed_skills_are_excluded(self):
        ordered = get_missing_skills_ordered(
            self.graph, possessed_skills={"python", "sql"}, required_skills={"ml", "data_modeling"}
        )
        # sql is possessed, but data_modeling (its dependent) is still needed
        self.assertEqual(set(ordered), {"statistics", "ml", "data_modeling"})

    def test_fully_possessed_goal_returns_empty_path(self):
        ordered = get_missing_skills_ordered(
            self.graph,
            possessed_skills={"python", "statistics", "sql", "ml", "data_modeling"},
            required_skills={"ml", "data_modeling"},
        )
        self.assertEqual(ordered, [])


if __name__ == "__main__":
    unittest.main()
