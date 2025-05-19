from flask import Flask, request, render_template
import os
import faiss
import numpy as np
import pdfplumber
from typing import List
from groq import Groq
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
chunks = []
index = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = None

# Load PDF
def load_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chunk text
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunk_list = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunk_list.append(chunk)
    return chunk_list

# Retrieve top-k similar chunks
def retrieve_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

# Generate answer using Groq
def generate_answer(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    global chunks, index, client

    answer = None
    error = None

    if request.method == "POST":
        try:
            api_key = request.form["api_key"]
            question = request.form["question"]
            uploaded_file = request.files["pdf_file"]

            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
                client = Groq(api_key=api_key)

            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)

                # Process PDF
                text = load_pdf_text(file_path)
                chunks = chunk_text(text)

                # Embeddings & FAISS index
                embeddings = embedder.encode(chunks, convert_to_tensor=True)
                embedding_dim = embeddings[0].shape[0]
                index = faiss.IndexFlatL2(embedding_dim)
                index.add(embeddings.cpu().detach().numpy())

            if question:
                answer = generate_answer(question)

        except Exception as e:
            error = str(e)

    return render_template("index.html", answer=answer, error=error)

if __name__ == "__main__":
    app.run(debug=True)
