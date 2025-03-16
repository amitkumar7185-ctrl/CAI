import os
import re
import math
import faiss
import pdfplumber
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import torch

# Cache models and heavy dependencies
#@st.cache_resource(show_spinner=False)
def initialize_dependencies():
    nltk.download('punkt')
    nltk.download('stopwords')
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    language_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return embedder, tokenizer, language_model

embedder, tokenizer, language_model = initialize_dependencies()

# ---------------------- Data Preprocessing ----------------------

def load_apple_filings_pdf(file_paths):
    texts = []
    for path in file_paths:
        all_text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"
                tables = page.extract_tables()
                for table in tables:
                    table_str = "\n".join([" | ".join([cell or "" for cell in row]) for row in table if row])
                    all_text += "\n[Extracted Table]\n" + table_str + "\n"
        texts.append(all_text)
    return texts

def chunk_text(text, chunk_size=250):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def preprocess_apple_data(file_paths):
    all_documents = []
    texts = load_apple_filings_pdf(file_paths)
    for i, text in enumerate(texts):
        file_name = os.path.basename(file_paths[i])
        chunks = chunk_text(text, chunk_size=250)
        for c in chunks:
            if len(c.strip()) > 0:
                all_documents.append({"text": c.strip(), "source": file_name})
    return all_documents

# ---------------------- FAISS ----------------------

def build_faiss_index(documents):
    doc_texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, doc_texts

# ---------------------- FAISS Retrieval ----------------------

def retrieve_documents_faiss(query, faiss_index, doc_texts, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    confidence_scores = 1 / (1 + distances)
    retrieved_results = [(doc_texts[idx], confidence_scores[0][i]) for i, idx in enumerate(indices[0])]
    return retrieved_results

# ---------------------- BM25 Fallback ----------------------

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def tokenize_for_bm25(text):
    tokens = re.findall(r"\w+", text.lower())
    tokens = [ps.stem(t) for t in tokens if t not in stop_words]
    return tokens

def bm25_score(query_tokens, doc_tokens, avg_doc_len, doc_freqs, total_docs):
    k1 = 1.5
    b = 0.75
    score = 0.0
    doc_len = len(doc_tokens)
    doc_term_counts = Counter(doc_tokens)
    for token in query_tokens:
        if token in doc_term_counts:
            df = doc_freqs.get(token, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            tf = doc_term_counts[token]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)
    return score

def retrieve_bm25(query, documents, top_k=3):
    query_tokens = tokenize_for_bm25(query)
    all_doc_tokens = [tokenize_for_bm25(doc["text"]) for doc in documents]
    doc_freqs = Counter()
    for dtoks in all_doc_tokens:
        for t in set(dtoks):
            doc_freqs[t] += 1
    total_docs = len(documents)
    avg_doc_len = sum(len(dtoks) for dtoks in all_doc_tokens) / total_docs
    scores = []
    for i, dtoks in enumerate(all_doc_tokens):
        score = bm25_score(query_tokens, dtoks, avg_doc_len, doc_freqs, total_docs)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = [(documents[idx]["text"], score) for idx, score in scores[:top_k]]
    return top_results

# ---------------------- Guardrails + LM ----------------------

def guardrail_input(user_input):
    if not user_input or len(user_input.strip()) == 0:
        return False, "Query cannot be empty."
    if len(user_input) > 1000:
        return False, "Query too long."
    return True, ""

def guardrail_output(response_text):
    if "I am not sure" in response_text:
        return "[Guardrail] The model seems uncertain. Please verify the response."
    return response_text

def generate_response(query, retrieved_docs, conversation_history):
    context_str = "\n\n".join(retrieved_docs)
    prompt = (
        f"User: {query}\n\n"
        f"Relevant Company Filings:\n{context_str}\n\n"
        f"Assistant:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = language_model.generate(
            input_ids,
            max_length=1500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return output_text.strip()
