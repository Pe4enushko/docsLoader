from __future__ import annotations

FIND_DOCUMENT_BY_HASH_SQL = """
SELECT id
FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'document'
  AND metadata->>'hash' = %s
LIMIT 1
"""

HYBRID_SEARCH_CHUNKS_SQL = """
SELECT metadata, (1 - (embedding <=> %s::vector)) AS score
FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'chunk'
ORDER BY score DESC
LIMIT %s
"""

FETCH_CHUNKS_BY_IDS_SQL = """
SELECT metadata
FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'chunk'
  AND metadata->>'chunk_id' = ANY(%s)
"""

FETCH_SECTION_NEIGHBORS_SQL = """
SELECT metadata
FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'chunk'
  AND metadata->>'section_path' = %s
ORDER BY ABS(((metadata->>'order')::int - %s))
LIMIT %s
"""

FETCH_CHUNKS_BY_ENTITY_MENTIONS_SQL = """
SELECT metadata
FROM graph_nodes
WHERE namespace = %s
  AND metadata->>'kind' = 'chunk'
  AND EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(COALESCE(metadata->'entity_mentions', '[]'::jsonb)) AS m(value)
    WHERE m.value = ANY(%s)
  )
LIMIT %s
"""

FETCH_CHUNKS_SUPPORTED_BY_RECOMMENDATIONS_SQL = """
WITH seed_chunks AS (
  SELECT id
  FROM graph_nodes
  WHERE namespace = %s
    AND metadata->>'kind' = 'chunk'
    AND metadata->>'chunk_id' = ANY(%s)
),
seed_recommendations AS (
  SELECT DISTINCT e.source_node_id AS rec_id
  FROM graph_edges e
  JOIN seed_chunks s ON s.id = e.target_node_id
  WHERE e.namespace = %s
    AND e.relation = 'supports_chunk'
)
SELECT DISTINCT c.metadata
FROM graph_edges e
JOIN seed_recommendations sr ON sr.rec_id = e.source_node_id
JOIN graph_nodes c ON c.id = e.target_node_id
WHERE e.namespace = %s
  AND e.relation = 'supports_chunk'
  AND c.metadata->>'kind' = 'chunk'
LIMIT %s
"""
