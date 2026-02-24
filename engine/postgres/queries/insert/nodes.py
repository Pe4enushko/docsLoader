from __future__ import annotations

UPSERT_NODE_SQL = """
INSERT INTO graph_nodes (namespace, content, embedding, metadata)
VALUES (%s, %s, %s::vector, %s)
ON CONFLICT (namespace, content) DO UPDATE SET
  embedding = EXCLUDED.embedding,
  metadata = graph_nodes.metadata || EXCLUDED.metadata
RETURNING id
"""

UPSERT_EDGE_SQL = """
INSERT INTO graph_edges (namespace, source_node_id, target_node_id, relation, weight, metadata)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (namespace, source_node_id, target_node_id, relation) DO UPDATE SET
  weight = EXCLUDED.weight,
  metadata = graph_edges.metadata || EXCLUDED.metadata
"""
