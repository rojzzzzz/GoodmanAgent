from dotenv import load_dotenv
from pgvector import psycopg

from config import DB_CREDENTIALS
load_dotenv()

def retrieve(query: str) -> list[dict[str, str | int | float]]:
    """Retrieves the most similar articles to the query from the database.
    Always use as the source of truth.
    
    Args:
        query (str): The query string to search for articles using semantic similarity.
    
    Returns:
        list: A list of the most similar articles to the query orderd by similarity score (closest first). Each article is represented as a dictionary with keys:
            - id: The unique identifier of the article.
            - source: The source of the article.
            - title: The title of the article.
            - number: The number of the article.
            - url: The URL of the article.
            - approval_date: The date the article was approved.
            - publication_date: The date the article was published.
            - distance: The similarity distance between the query and the article.
            - text: The text of the article."""

    from config import DB_CREDENTIALS, K_NEAREST_NEIGHBORS, EMBEDDING_MODEL
    from openai import OpenAI
    import psycopg

    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    with psycopg.connect(**DB_CREDENTIALS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    n.id, n.text, n.source_label,
                    i.titulo, i.numero, i.url, 
                    i.fecha_aprobacion, i.fecha_publicacion,
                    n.embedding <=> %(q)s::vector as distance
                FROM
                    document_nodes n
                JOIN
                    legal_instruments i
                    ON i.id=n.legal_instrument_id
                WHERE n.node_type = 'article'
                    AND i.extraction_status ='ok'
                    AND n.embedding IS NOT NULL
                ORDER BY n.embedding <=> %(q)s::vector, i.fecha_aprobacion DESC 
                LIMIT %(k)s
                """, {"q": query_embedding.data[0].embedding, "k": 20})

            results = cur.fetchall()

    results = [
                {
                   "id": row[0],
                    "source": row[2],
                    "title": row[3],
                    "number": row[4],
                    "url": row[5],
                    "approval_date": row[6],
                    "publication_date": row[7],
                    "distance": row[8],
                    "text": row[1]
                }
            for row in results]
    

    return results  

