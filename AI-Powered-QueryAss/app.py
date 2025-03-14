# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import psycopg2
# from transformers import pipeline
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI()

# # Load AI Model for Natural Language to SQL conversion
# # Use a model fine-tuned for SQL generation, e.g., "tscholak/cxmefzzi"
# try:
#     nlp_model = pipeline("text2text-generation", model="tscholak/cxmefzzi")
# except Exception as e:
#     raise RuntimeError(f"Failed to load the model: {e}")

# # Database connection details (Use environment variables)
# DB_CONFIG = {
#     "dbname": os.getenv("DB_NAME", "sqlgen"),
#     "user": os.getenv("DB_USER", "sqlgen_user"),
#     "password": os.getenv("DB_PASSWORD", "5GdmYbfyHKLIxR9MgRUF88wDWyFnqx8K"),
#     "host": os.getenv("DB_HOST", "dpg-cv8qkftumphs73csbqt0-a.oregon-postgres.render.com"),
#     "port": os.getenv("DB_PORT", "5432"),
# }

# # Pydantic model for request body
# class QueryRequest(BaseModel):
#     question: str

# # Function to execute SQL query
# def execute_sql_query(query: str):
#     try:
#         # Connect to the database
#         with psycopg2.connect(**DB_CONFIG) as conn:
#             with conn.cursor() as cursor:
#                 cursor.execute(query)
#                 result = cursor.fetchall()
#         return result
#     except psycopg2.Error as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

# # API endpoint to convert natural language to SQL
# @app.post("/generate_sql/")
# def generate_sql(request: QueryRequest):
#     question = request.question

#     # Generate SQL query using the model
#     try:
#         sql_query = nlp_model(question, max_length=100)[0]['generated_text']
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

#     # Execute SQL Query
#     try:
#         result = execute_sql_query(sql_query)
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error executing SQL query: {e}")

#     return {"sql_query": sql_query, "result": result}


import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import psycopg2

# Load fine-tuned SQL generation model
MODEL_NAME = "t5-small"  # Replace with fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

app = FastAPI()

# Database connection (Replace with your credentials)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "sqlgen"),
    "user": os.getenv("DB_USER", "sqlgen_user"),
    "password": os.getenv("DB_PASSWORD", "5GdmYbfyHKLIxR9MgRUF88wDWyFnqx8K"),
    "host": os.getenv("DB_HOST", "dpg-cv8qkftumphs73csbqt0-a.oregon-postgres.render.com"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Request model
class QueryRequest(BaseModel):
    question: str

# Convert natural language to SQL
def generate_sql(nl_query):
    input_text = f"Translate: {nl_query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100)
    sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql_query

# Execute SQL query (optional)
def execute_sql(query):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        conn.commit()
        conn.close()
        return result
    except Exception as e:
        return {"error": str(e)}

# API Route
@app.post("/generate_sql")
def get_sql(request: QueryRequest):
    sql_query = generate_sql(request.question)
    return {"sql": sql_query}

# API Route to execute SQL (optional)
@app.post("/execute_sql")
def run_sql(request: QueryRequest):
    sql_query = generate_sql(request.question)
    result = execute_sql(sql_query)
    return {"sql": sql_query, "result": result}

# # Run the app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

