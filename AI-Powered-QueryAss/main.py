from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model and tokenizer
MODEL_NAME = "Salesforce/codet5p-220m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

class QueryRequest(BaseModel):
    question: str

# def generate_sql(nl_query):
#     input_text = f"Translate to SQL: {nl_query}"
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#     output_ids = model.generate(input_ids)
#     sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return sql_query

def generate_sql(nl_query):
    input_text = f"Translate to SQL: {nl_query}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    output_ids = model.generate(input_ids, max_length=128)  # Ensure max length
    sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Remove extra tokens

    return sql_query.strip()  # Remove unwanted spaces

@app.post("/generate_sql")
async def generate_sql_endpoint(request: QueryRequest):
    sql_query = generate_sql(request.question)
    return {"sql": sql_query}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
