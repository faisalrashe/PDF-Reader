# from fastapi import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
import numpy as np
import re
import faiss
import os
from groq import Groq
import pickle

# model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
model_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
def load_file(file_path: str) -> str:

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
            
            
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,@-]', '', text)

    return text 


def text_to_chunks(text:str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 100)
    chunks = splitter.split_text(text)
    path = "faiss_db"
    os.makedirs(path , exist_ok=True)
    with open("faiss_db/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return chunks

def load_chunks():
    with open("faiss_db/chunks.pkl","rb") as f:
        return pickle.load(f)
    
    
def chunks_to_embeddings(chunks:list , model)-> list:
    emb = model.encode(chunks)
    emb = np.array(emb).astype("float32")
    return emb

def vector_store(emb:list):
    
    dimension = emb.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(emb)
    path = "faiss_db"
    os.makedirs(path , exist_ok=True)
    file_path = os.path.join(path, "index.faiss")
    faiss.write_index(index , file_path)
    # return index

def load_index():
    return faiss.read_index("faiss_db/index.faiss")

def query_processing(q:str , model):
    # chunks = text_to_chunks(q)
    emb = chunks_to_embeddings([q] , model)
    emb = np.array(emb).astype("float32")
    return emb

def search(q:str , model , k=7):
    chunks = load_chunks()
    index = load_index()
    q_emb = query_processing(q ,model)
    
    distance , indices = index.search(q_emb , k)
    
    result = [chunks[i] for i in indices[0]]
    
    return result

def generate_response(chunks:list , question : str) -> str:
    prompt = f"""
        You are a helpful AI assistant.

        Your task is to answer the user's question ONLY from the provided context.

        Instructions:
        - Use only the given context.
        - If the answer is not available in the context, say:
        "The information is not available in the provided document."
        - Keep the answer clear and concise.
        - Do not make up information.
        - Summarize the answer in a human-friendly way.

        Context:
        {chunks}

        Question:
        {question}

        Answer:
        """
    response = model_llm.chat.completions.create(model="llama-3.3-70b-versatile",
                                                 temperature=0.5,
                                                 messages=[{"role":"system",
                                                           "content":prompt}])
    
    answer = response.choices[0].message.content
    return answer

if __name__ =="__main__":
    text = load_file("./uploads/data.pdf")

    chunks = text_to_chunks(text)

    embeddings = chunks_to_embeddings(chunks)

    vector_store(embeddings)
    
    q = "Who is the Founder of Pakistan"
    
    result = search(q)
    
    # for i in result:
    #     print(i)
    
    answer = generate_response(result , q)
    
    print(answer)
    
    
