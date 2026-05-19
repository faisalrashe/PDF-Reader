from fastapi import FastAPI , UploadFile , File , Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import os
import shutil
from services import *


app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
templates = Jinja2Templates(directory="frontend")

class AskQuestion(BaseModel):
    question:str

@app.get("/" , response_class=HTMLResponse)
def home(request : Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request
        )
@app.post("/question")
def form_data(data:AskQuestion):
    result = search(data.question , model)
    answer = generate_response(result , data.question)
    return {
        "question":data.question,
        "answer": answer,
    }
    

@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    file_path = f"uploads/data.pdf"
    
    os.makedirs("uploads" , exist_ok=True)
    # os.makedirs("fiass", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    text = load_file(file_path=file_path)

    chunks = text_to_chunks(text)

    embeddings = chunks_to_embeddings(chunks , model)

    vector_store(embeddings)
        
        
    return {"message" : "File Upload Successfully",
            "File Path": file_path,}