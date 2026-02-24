from __future__ import annotations

DELETE_DOCUMENT_EDGES_SQL = """
DELETE FROM graph_edges WHERE namespace = %s
"""

DELETE_DOCUMENT_NODES_SQL = """
DELETE FROM graph_nodes WHERE namespace = %s
"""

DELETE_DOCUMENT_META_NODE_SQL = """
DELETE FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'document'
  AND metadata->>'doc_id' = %s
"""
