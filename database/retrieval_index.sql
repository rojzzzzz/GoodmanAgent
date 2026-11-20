-- Apply after loading article embeddings. Run outside a transaction block.
CREATE INDEX CONCURRENTLY IF NOT EXISTS document_nodes_embedding_hamming_idx
    ON document_nodes USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops)
    WHERE node_type = 'article' AND embedding IS NOT NULL;
