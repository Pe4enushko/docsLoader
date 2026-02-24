from __future__ import annotations

DELETE_NAMESPACE_GRAPH_SQL = """
DELETE FROM graph_edges WHERE namespace = %s;
DELETE FROM graph_nodes WHERE namespace = %s;
"""

DELETE_ALL_GRAPH_SQL = """
DELETE FROM graph_edges;
DELETE FROM graph_nodes;
"""
