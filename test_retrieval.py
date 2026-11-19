from config import DB_CREDENTIALS, K_NEAREST_NEIGHBORS
from dotenv import load_dotenv
from openai import OpenAI
import psycopg
import os

load_dotenv()

client = OpenAI()
MODEL = "text-embedding-3-small"

query = "Mi vecino hace ruido por las noches"
query_embedding = client.embeddings.create(
    model=MODEL,
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
position = 1
for row in results:
    print(f"Position: {position}")
    print(f"ID: {row[0]}, Source: {row[2]}, Title: {row[3]}, Number: {row[4]}, URL: {row[5]}, Approval Date: {row[6]}, Publication Date: {row[7]}, Distance: {row[8]}")
    print(f"Text: {row[1]}")
    print()
    position += 1