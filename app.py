from src.helper import download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from src.prompt import *
import os
from flask import Flask, render_template, jsonify, request
embedding_model =SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)

PINECONE_API_KEY="60a9a852-6ad2-4041-9ddf-b4b2d2d06731"
PINECONE_API_ENV="us-east-1"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('medical-bot')



llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type='llama',
                  config={'max_new_tokens':512,
                          'temperature':0.8})

def extract_text_from_results(results):
    extracted_texts = []
    for match in results['matches']:
        extracted_texts.append(match['metadata']['text'])  # Assuming 'text' field in metadata
    return " ".join(extracted_texts) 
@app.route("/")
def index1():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    query1=request.form['msg']
    print(query1)
    query_vector = embedding_model.encode(query1)
    results = index.query(vector=query_vector, top_k=3,include_metadata=True)
    context = extract_text_from_results(results)
    prompt = prompt_template.format(context=context, question=query1)
    response = llm.generate([prompt])
    text = response.generations[0][0].text
    print(text)
    return str(text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)