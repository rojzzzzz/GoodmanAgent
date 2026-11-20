BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS legal_instruments (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source TEXT NOT NULL CHECK (source = 'digesto'),
    source_id TEXT NOT NULL CHECK (btrim(source_id) <> '' AND source_id = btrim(source_id)),
    categoria TEXT,
    numero TEXT,
    rango TEXT,
    titulo TEXT NOT NULL CHECK (btrim(titulo) <> ''),
    fecha_aprobacion DATE,
    fecha_publicacion DATE,
    url TEXT NOT NULL CHECK (btrim(url) <> ''),
    metadata_fetched_at TIMESTAMPTZ NOT NULL,
    text TEXT,
    text_sha256 TEXT CHECK (text_sha256 IS NULL OR text_sha256 ~ '^[0-9a-f]{64}$'),
    text_chars INTEGER CHECK (text_chars IS NULL OR text_chars >= 0),
    extraction_status TEXT NOT NULL CHECK (extraction_status IN (
        'not_attempted', 'ok', 'empty_response', 'no_usable_text', 'needs_review', 'http_error'
    )),
    status_reason TEXT,
    text_fetched_at TIMESTAMPTZ,
    extractor_version TEXT CHECK (extractor_version IS NULL OR btrim(extractor_version) <> ''),
    CONSTRAINT instrument_source_unique UNIQUE (source, source_id),
    -- Explicit null guards avoid CHECK's acceptance of SQL UNKNOWN.
    CONSTRAINT instrument_content_matches_status CHECK (
        (
            extraction_status = 'not_attempted'
            AND text IS NULL AND text_sha256 IS NULL AND text_chars IS NULL
            AND status_reason IS NULL AND text_fetched_at IS NULL AND extractor_version IS NULL
        )
        OR
        (
            extraction_status = 'ok'
            AND text IS NOT NULL AND btrim(text) <> ''
            AND text_sha256 IS NOT NULL
            AND text_chars IS NOT NULL AND text_chars = char_length(text)
            AND text_fetched_at IS NOT NULL AND extractor_version IS NOT NULL
        )
        OR
        (
            extraction_status IN ('empty_response', 'no_usable_text', 'needs_review', 'http_error')
            AND text IS NOT NULL AND text = '' AND text_sha256 IS NULL
            AND text_chars IS NOT NULL AND text_chars = 0
            AND text_fetched_at IS NOT NULL AND extractor_version IS NOT NULL
            AND status_reason IS NOT NULL AND btrim(status_reason) <> ''
        )
    )
);

CREATE TABLE IF NOT EXISTS document_nodes (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- Loader resolves the transformed source_id to this generated database ID.
    legal_instrument_id BIGINT NOT NULL
        REFERENCES legal_instruments(id) ON DELETE CASCADE,
    node_key TEXT NOT NULL CHECK (btrim(node_key) <> ''),
    parent_node_key TEXT CHECK (parent_node_key IS NULL OR btrim(parent_node_key) <> ''),
    position INTEGER NOT NULL CHECK (position >= 0),
    node_type TEXT NOT NULL CHECK (node_type IN (
        'article', 'paragraph', 'heading', 'annex', 'list_item',
        'table', 'preface', 'preamble', 'conclusion'
    )),
    label TEXT CHECK (label IS NULL OR btrim(label) <> ''),
    source_label TEXT,
    text TEXT NOT NULL CHECK (btrim(text) <> ''),
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL,
    source_locator TEXT NOT NULL CHECK (btrim(source_locator) <> ''),
    links JSONB NOT NULL DEFAULT '[]'::JSONB CHECK (jsonb_typeof(links) = 'array'),
    attributes JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (jsonb_typeof(attributes) = 'object'),

    embedding VECTOR(1536),

    CONSTRAINT node_key_per_instrument UNIQUE (legal_instrument_id, node_key),
    CONSTRAINT node_position_per_instrument UNIQUE (legal_instrument_id, position),
    CONSTRAINT valid_node_span CHECK (
        end_char > start_char AND char_length(text) = end_char - start_char
    ),
    CONSTRAINT node_not_own_parent CHECK (
        parent_node_key IS NULL OR parent_node_key <> node_key
    ),
    CONSTRAINT parent_in_same_instrument
        FOREIGN KEY (legal_instrument_id, parent_node_key)
        REFERENCES document_nodes(legal_instrument_id, node_key)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
);

-- Upgrade databases created before article embeddings were introduced.
ALTER TABLE document_nodes ADD COLUMN IF NOT EXISTS embedding VECTOR(1536);

CREATE INDEX IF NOT EXISTS document_nodes_parent_idx
    ON document_nodes (legal_instrument_id, parent_node_key);

CREATE INDEX IF NOT EXISTS document_nodes_article_text_idx
    ON document_nodes USING GIN (to_tsvector('pg_catalog.spanish', text))
    WHERE node_type = 'article';

CREATE INDEX IF NOT EXISTS legal_instruments_title_search_idx
    ON legal_instruments USING GIN (to_tsvector('pg_catalog.spanish', titulo))
    WHERE extraction_status = 'ok';

CREATE INDEX IF NOT EXISTS legal_instruments_number_idx
    ON legal_instruments (numero)
    WHERE extraction_status = 'ok';

COMMIT;
