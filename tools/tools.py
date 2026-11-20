"""Retrieve source articles from the Digesto corpus."""

from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai import Agent, UsageLimits
from openai import OpenAI, OpenAIError
from psycopg.rows import dict_row
from dotenv import load_dotenv
from typing import TypedDict
from datetime import date
from typing import Any
import unicodedata
import logging
import re




import psycopg



from pipelines.initial.digesto_document import ARTICLE_NUMBER

logger = logging.getLogger(__name__)

class VerificationResult(TypedDict):
    is_relevant: bool
    explanation:str

def retrieve(
    query: str, *, limit: int | None = None, semantic: bool = True
) -> list[dict[str, str | int | float | date | None]]:
    """Find Digesto articles and source metadata for a Spanish query.

    A single explicit citation is looked up by instrument and article number.
    Other queries combine rankings from article text, instrument titles, and
    cosine distance. Set semantic=False to use local text search. If query
    embeddings are unavailable, text search still works. Returned article
    text is verbatim. The distance is present when both vectors exist, and
    score is the fused rank score for a topical query.
    """
    import config
    import psycopg

    _VECTOR_SHORTLIST_SIZE = 1000
    _VECTOR_INDEX = "document_nodes_embedding_hamming_idx"
    _VECTOR_INDEX_STATUS_SQL = """
        SELECT EXISTS (
            SELECT FROM pg_index
            WHERE indexrelid = to_regclass(%s) AND indrelid = to_regclass('document_nodes')
            AND indisvalid
        ) AS ready
    """

    _INSTRUMENT = (
        r"(?P<kind>ley|c[oó]digo|decreto(?:[- ]ley| (?:ejecutivo|legislativo|jgrn|jng))?"
        r"|acuerdo(?: (?:ejecutivo|presidencial|ministerial|administrativo|jgrn|jng))?"
        r"|resoluci[oó]n(?: (?:administrativa|legislativa|ministerial))?)\s+"
        r"(?:(?:n[.°º]+|no\.?|n[úu]m(?:ero)?\.?)\s*)?"
        r"(?P<number>[0-9]+(?:\s*[-–—/]\s*[0-9]+)*)"
    )
    _ARTICLE = rf"(?:art[íi]culo|arto?\.?)\s*{ARTICLE_NUMBER}"
    _LOOKUP_PREFIX = r"(?:(?:qu[eé] (?:dice|establece)|cita|mu[eé]strame)\s+)?(?:el |la )?"
    _CITATIONS = tuple(re.compile(_LOOKUP_PREFIX + pattern, re.IGNORECASE) for pattern in (
        rf"{_ARTICLE}\s+(?:de la |del |de )?{_INSTRUMENT}",
        rf"{_INSTRUMENT}(?:(?:\s*[,;:]\s*|\s+){_ARTICLE})?",
    ))

    _RESULT_COLUMNS = """
        n.id, n.source_label AS source, i.titulo AS title, i.numero AS number, i.url,
        i.fecha_aprobacion AS approval_date, i.fecha_publicacion AS publication_date,
        n.text, n.label AS article_label, i.source_id, n.node_key
    """

    _CITATION_SQL = rf"""
        SELECT {_RESULT_COLUMNS}, NULL::float8 AS distance, NULL::float8 AS score
        FROM legal_instruments i
        JOIN document_nodes n ON n.legal_instrument_id = i.id
        WHERE i.extraction_status = 'ok'
        AND n.node_type = 'article'
        AND i.numero = %(number)s
        AND CASE
            WHEN %(kind)s = 'ley' THEN lower(i.rango) IN ('ley', 'código')
            WHEN %(kind)s IN ('decreto', 'acuerdo', 'resolución')
                THEN lower(i.rango) ~ ('^' || %(kind)s || '([ -]|$)')
            ELSE lower(i.rango) = %(kind)s
        END
        AND (%(article)s::text IS NULL OR
            translate(
                regexp_replace(lower(n.label), '([0-9])[º°o]|[[:space:]]+', '\1', 'g'),
                'úá', 'ua'
            ) = %(article)s)
        ORDER BY i.fecha_aprobacion DESC NULLS LAST, i.id, n.position
        LIMIT %(limit)s
    """

    _EXACT_SEMANTIC_SQL = """
        SELECT n.id, n.embedding <=> %(embedding)s::vector AS distance
        FROM document_nodes n
        JOIN legal_instruments i ON i.id = n.legal_instrument_id
        WHERE %(embedding)s::vector IS NOT NULL
        AND n.node_type = 'article' AND n.embedding IS NOT NULL
        AND i.extraction_status = 'ok'
        ORDER BY distance, n.id
        LIMIT %(candidates)s
    """

    _INDEXED_SEMANTIC_SQL = f"""
        WITH nearest AS MATERIALIZED (
            SELECT n.id
            FROM document_nodes n
            WHERE n.node_type = 'article' AND n.embedding IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM legal_instruments i
                WHERE i.id = n.legal_instrument_id AND i.extraction_status = 'ok'
            )
            ORDER BY binary_quantize(n.embedding)::bit({config.EMBEDDING_DIMENSIONS})
                <~> binary_quantize(%(embedding)s::vector)::bit({config.EMBEDDING_DIMENSIONS})
            LIMIT %(shortlist)s
        )
        SELECT n.id, n.embedding <=> %(embedding)s::vector AS distance
        FROM nearest JOIN document_nodes n USING (id)
        ORDER BY distance, n.id
        LIMIT %(candidates)s
    """

    _HYBRID_SQL = """
        WITH raw_query AS (
            SELECT plainto_tsquery('pg_catalog.spanish', %(query)s) AS all_terms,
                phraseto_tsquery('pg_catalog.spanish', %(query)s) AS phrase
        ),
        query_terms AS (
            SELECT q.all_terms, q.phrase, replace(q.all_terms::text, ' & ', ' | ')::tsquery AS any_term,
                coalesce(pairs.query, q.all_terms) AS match_terms
            FROM raw_query q
            LEFT JOIN LATERAL (
                -- For three to seven terms, prepare a fallback with at least two distinct terms.
                SELECT string_agg('(' || a.term || ' & ' || b.term || ')', ' | ')::tsquery AS query
                FROM unnest(string_to_array(q.all_terms::text, ' & ')) a(term)
                JOIN unnest(string_to_array(q.all_terms::text, ' & ')) b(term) ON a.term < b.term
                WHERE numnode(q.all_terms) BETWEEN 5 AND 13
            ) pairs ON true
        ),
        semantic AS ({semantic_sql}),
        matching_titles AS (
            SELECT i.id, lower(i.titulo) = lower(%(query)s) AS exact_title,
                ts_rank_cd(to_tsvector('pg_catalog.spanish', i.titulo), q.all_terms, 1) AS relevance
            FROM legal_instruments i
            CROSS JOIN query_terms q
            WHERE i.extraction_status = 'ok'
            AND to_tsvector('pg_catalog.spanish', i.titulo) @@ q.all_terms
            AND EXISTS (
                SELECT 1 FROM document_nodes n
                WHERE n.legal_instrument_id = i.id AND n.node_type = 'article'
            )
            ORDER BY exact_title DESC, relevance DESC, i.id
            LIMIT %(candidates)s
        ),
        body_query AS MATERIALIZED (
            SELECT q.any_term, q.phrase, CASE
                WHEN q.match_terms = q.all_terms OR EXISTS (SELECT FROM matching_titles)
                    OR EXISTS (
                        SELECT FROM document_nodes n
                        JOIN legal_instruments i ON i.id = n.legal_instrument_id
                        WHERE n.node_type = 'article' AND i.extraction_status = 'ok'
                        AND to_tsvector('pg_catalog.spanish', n.text) @@ q.all_terms
                    )
                THEN q.all_terms
                ELSE q.match_terms
            END AS match_terms
            FROM query_terms q
        ),
        matching_articles AS MATERIALIZED (
            SELECT n.id, to_tsvector('pg_catalog.spanish', n.text) AS terms
            FROM document_nodes n
            JOIN legal_instruments i ON i.id = n.legal_instrument_id
            CROSS JOIN body_query q
            WHERE n.node_type = 'article' AND i.extraction_status = 'ok'
            AND to_tsvector('pg_catalog.spanish', n.text) @@ q.match_terms
        ),
        article_text AS (
            SELECT n.id, n.terms @@ q.phrase AS phrase_match, ts_rank(n.terms, q.any_term, 1) AS relevance
            FROM matching_articles n
            CROSS JOIN body_query q
            ORDER BY phrase_match DESC, relevance DESC, n.id
            LIMIT %(candidates)s
        ),
        instrument_title AS (
            SELECT n.id, t.exact_title, t.relevance, n.body_relevance, t.id AS instrument_id, n.position
            FROM matching_titles t
            CROSS JOIN query_terms q
            CROSS JOIN LATERAL (
                SELECT n.id, n.position,
                    ts_rank(to_tsvector('pg_catalog.spanish', n.text), q.any_term, 1) AS body_relevance
                FROM document_nodes n
                WHERE n.legal_instrument_id = t.id AND n.node_type = 'article'
                ORDER BY body_relevance DESC, n.position
                LIMIT 3
            ) n
            ORDER BY t.exact_title DESC, t.relevance DESC, n.body_relevance DESC, t.id, n.position
            LIMIT %(candidates)s
        ),
        ranked AS (
            SELECT id, row_number() OVER (ORDER BY distance, id) AS rank, 1.0 AS weight FROM semantic
            UNION ALL
            SELECT id, row_number() OVER (ORDER BY phrase_match DESC, relevance DESC, id),
                CASE WHEN phrase_match THEN 0.65 ELSE 0.12 END
            FROM article_text
            UNION ALL
            SELECT id, row_number() OVER (
                ORDER BY exact_title DESC, relevance DESC, body_relevance DESC, instrument_id, position
            ), CASE WHEN exact_title THEN 2.0 ELSE 0.5 END FROM instrument_title
        ),
        fused AS (
            -- Loose term matches carry less evidence than a phrase or an exact title.
            SELECT id, sum(weight / (8 + rank))::float8 AS score
            FROM ranked
            GROUP BY id
            ORDER BY score DESC, id
            LIMIT %(limit)s
        )
        SELECT {result_columns}, n.embedding <=> %(embedding)s::vector AS distance, f.score
        FROM fused f
        JOIN document_nodes n ON n.id = f.id
        JOIN legal_instruments i ON i.id = n.legal_instrument_id
        ORDER BY f.score DESC, n.id
    """

    _HYBRID_EXACT_SQL = _HYBRID_SQL.format(semantic_sql=_EXACT_SEMANTIC_SQL, result_columns=_RESULT_COLUMNS)
    _HYBRID_INDEXED_SQL = _HYBRID_SQL.format(semantic_sql=_INDEXED_SEMANTIC_SQL, result_columns=_RESULT_COLUMNS)


    def parse_citation(query: str) -> dict[str, str | None] | None:
        """Recognize a single numbered citation expressed as a lookup."""
        query = re.sub(r"(?<=[0-9])\.(?=[º°])", "", query.strip("¿?¡!. "))
        for pattern in _CITATIONS:
            match = pattern.fullmatch(query)
            if match:
                kind = match["kind"].lower().replace("decreto ley", "decreto-ley")
                kind = {"codigo": "código", "resolucion": "resolución"}.get(kind, kind)
                article = match["label"]
                if article is not None:
                    article = re.sub(r"([0-9])[º°o]|\s+", r"\1", article.lower())
                    article = article.replace("ú", "u").replace("á", "a")
                number = re.sub(r"\s+", "", match["number"]).replace("–", "-").replace("—", "-")
                return {"kind": kind, "number": number, "article": article}
        return None

    def embed_query(query: str) -> str | None:
        from pgvector import Vector

        load_dotenv(config.ROOT_DIR / ".env")
        try:
            with OpenAI(timeout=8, max_retries=0) as client:
                response = client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    dimensions=config.EMBEDDING_DIMENSIONS,
                    input=query,
                )
        except OpenAIError as error:
            logger.warning("Query embedding unavailable (%s); using text search", type(error).__name__)
            return None
        return Vector(response.data[0].embedding).to_text()

    def has_vector_index(connection: psycopg.Connection) -> bool:
        row = connection.execute(_VECTOR_INDEX_STATUS_SQL, (_VECTOR_INDEX,)).fetchone()
        return row["ready"]

    if limit is None:
        limit = config.K_NEAREST_NEIGHBORS
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")

    query = " ".join(unicodedata.normalize("NFC", query).split())
    if not any(character.isalnum() for character in query):
        return []

    citation = parse_citation(query)
    if citation is not None:
        statement = _CITATION_SQL
        params = {**citation, "limit": limit}
    else:
        statement = _HYBRID_EXACT_SQL
        params = {
            "query": query,
            "embedding": embed_query(query) if semantic else None,
            "candidates": limit * 4,
            "limit": limit,
        }

    with psycopg.connect(**config.DB_CREDENTIALS, row_factory=dict_row) as connection:
        if (citation is None and params["embedding"] is not None
                and params["candidates"] < _VECTOR_SHORTLIST_SIZE and has_vector_index(connection)):
            connection.execute("SELECT set_config('hnsw.ef_search', %s, true)",
                               (str(_VECTOR_SHORTLIST_SIZE),))
            statement = _HYBRID_INDEXED_SQL
            params["shortlist"] = _VECTOR_SHORTLIST_SIZE
        return connection.execute(statement, params).fetchall()

def get_article_context(source_id: str, node_key: str):
    """Reads the surrounding articles and section headings from the same instrument.
        Args:
            source_id (str), node_key (str)"""
    
    import psycopg
    import config

    with psycopg.connect(**config.DB_CREDENTIALS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    i.categoria, i.rango,
                    n.node_type, n.label, n.source_label,
                    
                FROM document_nodes n
                JOIN legal_instruments i ON n.legal_instrument_id = i.id
                WHERE i.source_id = %s
            """, (source_id))
            return cur.fetchall()

def calculator(a: float, b: float, operation: str) -> float:
    """Performs basic arithmetic operations on two numbers.
    
    Args:
        a (float): The first number.
        b (float): The second number.
        operation (str): The operation to perform. Can be 'add', 'subtract', 'multiply', or 'divide'.
    Returns:
        float: The result of the arithmetic operation.
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b
    else:
        raise ValueError("Invalid operation. Please use 'add', 'subtract', 'multiply', or 'divide'.")

def verify_response(response: str, query: str) -> dict[str, Any]:
    """Deplos a subagent that verifies if the response is relevant to the query
    
    Args:
        response (str): The response to verify.
        query (str): The query to check against.

    Returns:
        dict with keys: 'is_relevant' (bool) and 'explanation' (str)
    """
    from pydantic_ai import Agent, UsageLimits
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file

    agent = Agent(
        "openai:gpt-6-luna",
        instructions="""You are a verification agent. Your task is to determine if the provided response is relevant to the given query""",
        output_type=VerificationResult
    )

    result = agent.run_sync(f"Query: {query}\nResponse: {response}\nIs the response relevant to the query? Please provide a boolean answer and a brief explanation.",
                            usage_limits=UsageLimits(total_tokens_limit=1200, request_limit=4))

    return result.output

def verify_vigencia(source_id: str, article: str | None = None) -> dict[str, str | None]:
    """Check an instrument's registry classification in the official Nicaraguan Digesto.

    Use this before relying on an instrument whose current status is uncertain.
    Pass the encoded source_id returned by retrieve and, optionally, an article label.
    The record supports an instrument-level finding. For a specific article, use
    instrument_status as context and seek provision-specific evidence separately.
    checked_at records the lookup time; it is not the legal effective date.
    """
    from html import unescape

    from bs4 import BeautifulSoup
    import httpx

    from pipelines.initial.extract import is_access_challenge, is_real_record, source_url, utc_now

    def record_values(script: str, field: str) -> set[str]:
        pattern = rf"""^[ \t]*var[ \t]+{field}[ \t]*=[ \t]*(['"])(.*?)\1[ \t]*;[ \t]*\r?$"""
        return {" ".join(unescape(value).split()) for _, value in re.findall(pattern, script, re.MULTILINE)}

    source_id = source_id.strip()
    article = (article or "").strip() or None
    result: dict[str, str | None] = {
        "source_id": source_id,
        "article": article,
        "status": "unresolved",
        "instrument_status": "unresolved",
        "classification": None,
        "source_url": None,
        "checked_at": None,
        "explanation": "Use the encoded Digesto source_id returned by retrieve.",
    }
    if len(source_id) > 128 or not is_real_record({"iunpid": source_id}):
        return result

    url = source_url(source_id)
    result.update(source_url=url, checked_at=utc_now())
    try:
        response = httpx.get(url, timeout=8, follow_redirects=False)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        result["explanation"] = f"The Digesto record returned HTTP {error.response.status_code}."
        return result
    except httpx.RequestError:
        result["explanation"] = "The official Digesto record could not be reached."
        return result

    if is_access_challenge(response.content):
        result["explanation"] = "The Digesto returned an access challenge."
        return result

    page = BeautifulSoup(response.content, "html.parser")
    script = "\n".join(tag.get_text() for tag in page.find_all("script", src=False))
    classifications = record_values(script, "registroNorma")
    if record_values(script, "idNorma") != {source_id} or len(classifications) != 1:
        result["explanation"] = "The Digesto response did not identify a matching record with one classification."
        return result

    classification = classifications.pop()
    instrument_status = {
        "vigente": "in_force",
        "consolidada": "in_force",
        "sin vigencia o derecho histórico": "not_in_force",
    }.get(classification.casefold(), "unresolved")
    result.update(classification=classification or None, instrument_status=instrument_status)

    if instrument_status == "unresolved":
        result["explanation"] = "The Digesto record did not provide a recognized vigencia classification."
    elif article is not None:
        result["explanation"] = "The registry classifies the instrument. The requested article needs separate evidence."
    else:
        result.update(status=instrument_status, explanation="Status reported by the official Digesto registry.")
    return result


TOOLS = [retrieve, web_fetch_tool, calculator, verify_response, get_article_context, verify_vigencia,
         duckduckgo_search_tool()]
