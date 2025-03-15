import os
import re
import math
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import faiss
from collections import Counter


import sys
print(sys.path)

import subprocess
installed_packages = subprocess.run(["pip", "list"], capture_output=True, text=True)
print(installed_packages.stdout)


# Download NLTK resources (only needed once)
#nltk.data.path = [r'C:\Users\hemam\nltk_data']
# nltk.download('punkt', download_dir=r'C:\Users\hemam\nltk_data')
# nltk.download('punkt_tab', download_dir=r'C:\Users\hemam\nltk_data')
# nltk.download('stopwords', download_dir=r'C:\Users\hemam\nltk_data')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

###############################################################################
# 1. DATA COLLECTION & PREPROCESSING
###############################################################################

def load_apple_filings(file_paths):
    """
    Load text from Apple 10-K/10-Q filings provided as file paths.
    Returns a list of texts.
    """
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            texts.append(text)
    return texts

def chunk_text(text, chunk_size=250):
    """
    Splits text into smaller chunks (by sentence) with a maximum of ~chunk_size tokens.   
    """
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
    """
    Reads Apple filings from given file paths, chunks them,
    and returns a list of document dictionaries.
    Each dictionary contains:
      - "text": the chunk text
      - "source": the file name it came from
    """
    all_documents = []
    texts = load_apple_filings(file_paths)
    for i, text in enumerate(texts):
        file_name = os.path.basename(file_paths[i])
        chunks = chunk_text(text, chunk_size=200)
        for c in chunks:
            if len(c.strip()) > 0:
                all_documents.append({
                    "text": c.strip(),
                    "source": file_name
                })
    return all_documents

###############################################################################
# 2. SETUP EMBEDDING MODEL, LANGUAGE MODEL & FAISS INDEX
###############################################################################

# 2.1 Initialize the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)

# 2.2 Initialize the language model (small open-source LM)
lm_model_name = "distilGPT2"  # or you can use "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

# 2.3 Global variables to store FAISS index and document texts
faiss_index = None
doc_texts = []  # list of chunk texts
doc_metadatas = []  # corresponding metadata (e.g., source)

def build_faiss_index(documents):
    """
    Build a FAISS index from the document chunks.
    This function creates the embeddings for each document chunk,
    builds the index, and stores global variables for retrieval.
    """
    global faiss_index, doc_texts, doc_metadatas

    doc_texts = [doc["text"] for doc in documents]
    doc_metadatas = [doc["source"] for doc in documents]
    embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

    # Determine embedding dimension and build index
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)

    return faiss_index

###############################################################################
# 3. ADVANCED RETRIEVAL IMPLEMENTATION (WITH FAISS & Optional BM25)
###############################################################################

def retrieve_documents_faiss(query, top_k=3):
    """
    Retrieve top_k relevant document chunks from FAISS using the query.
    Returns retrieved texts along with their confidence scores.
    """
    global faiss_index, doc_texts
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Convert distances to similarity scores (higher score is better)
    confidence_scores = 1 / (1 + distances)

    retrieved_results = [(doc_texts[idx], confidence_scores[0][i]) for i, idx in enumerate(indices[0])]
    
    return retrieved_results  # Return a list of tuples (document, confidence score)


# BM25 fallback implementation

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
    
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by highest score
    top_results = [(documents[idx]["text"], score) for idx, score in scores[:top_k]]
    
    return top_results  # Return (document, confidence score)

###############################################################################
# 4. GUARDRAILS & RESPONSE GENERATION
###############################################################################

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
    """
    Generate a response using the small language model.
    The prompt includes (optionally) a small conversation history and the retrieved context.
    """
    context_str = "\n\n".join(retrieved_docs)
    history_str = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in conversation_history[-3:]]
    )

    prompt = (
        f"User: {query}\n\n"
        f"Relevant Company Filings:\n{context_str}\n\n"
        f"Assistant:"
    )

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = language_model.generate(
            input_ids,
            max_length=1000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return output_text.strip()

###############################################################################
# 5. UI DEVELOPMENT (STREAMLIT)
###############################################################################

import glob
import os

def get_all_txt_files(folder_path):
    """
    Retrieve all .txt file paths from the given folder.
    """
    return glob.glob(os.path.join(folder_path, "*.txt"))

def main():
    st.title("Apple Inc. Financial RAG Chatbot with FAISS")

    # Define your folder path
    folder_path = r"Data"

    # Get all .txt files from the folder
    file_paths = get_all_txt_files(folder_path)

    # file_paths = []
    # file_path = r'D:\Study\Code_py\cai\temp_Apple_10K_2022.txt'
    # file_paths.append(file_path)
    documents = preprocess_apple_data(file_paths)
    print(f"Total chunks created: {len(documents)}")
    build_faiss_index(documents)
    st.session_state.documents = documents
    st.success("Indexes built successfully.")


    # Initialize session state for conversation and document storage
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = None

    # st.subheader("1. Upload Apple Filings (Text Files)")
    # st.write("Upload one or more Apple 10-K/10-Q text files for indexing.")
    # uploaded_files = st.file_uploader("Upload Apple filings (.txt)", type=["txt"], accept_multiple_files=True)

    # if uploaded_files:
    #     file_paths = []
    #     for uploaded_file in uploaded_files:
    #         temp_path = os.path.join("temp_" + uploaded_file.name)
    #         with open(temp_path, "wb") as f:
    #             f.write(uploaded_file.read())
    #         file_paths.append(temp_path)

    #     if st.button("Build FAISS Index"):
    #         documents = preprocess_apple_data(file_paths)
    #         st.write(f"Total chunks created: {len(documents)}")
    #         build_faiss_index(documents)
    #         st.session_state.documents = documents  # store for potential BM25 fallback
    #         st.success("FAISS index built successfully.")

    st.subheader("1. Ask a Question")
    user_query = st.text_input("Enter your question about Apple's financials")

    if st.button("Submit Query"):
        is_valid, msg = guardrail_input(user_query)
        if not is_valid:
            st.error(msg)
        else:
            # Retrieve documents via FAISS
            retrieved_docs_with_scores = retrieve_documents_faiss(user_query, top_k=3)

            # If FAISS returns no documents, use BM25 as a fallback
            if not retrieved_docs_with_scores or len(retrieved_docs_with_scores) == 0:
                st.warning("No documents found with FAISS. Trying BM25 fallback...")
                if st.session_state.documents:
                    retrieved_docs_with_scores = retrieve_bm25(user_query, st.session_state.documents, top_k=3)
                else:
                    retrieved_docs_with_scores = []

            # Generate response from the language model
            retrieved_texts = [doc for doc, score in retrieved_docs_with_scores]
            response_text = generate_response(user_query, retrieved_texts, st.session_state.conversation_history)
            response_text = guardrail_output(response_text)

            # Store conversation history with confidence scores
            st.session_state.conversation_history.append({
                "user": user_query,
                "retrieved_docs": retrieved_docs_with_scores,  # Store docs & scores
                "assistant": response_text
            })

            # Display retrieved documents with confidence scores
            st.subheader("Retrieved Documents with Confidence Scores:")
            for i, (doc, score) in enumerate(retrieved_docs_with_scores):
                st.write(f"**Document {i+1}:**")
                st.write(f"**Confidence Score:** {score:.4f}")
                st.write(doc)            
      

    # Display conversation history with confidence scores
    st.subheader("Conversation History")
    for turn in st.session_state.conversation_history:
        st.write(f"**User:** {turn['user']}")
        for i, (doc, score) in enumerate(turn['retrieved_docs']):
            st.write(f"**Retrieved Doc {i+1} Confidence Score:** {score:.4f}")
        st.write(f"**Assistant:** {turn['assistant']}")

    st.subheader("3. Testing & Validation")
    st.write("Try queries like:")
    st.write("- 'What is Apple’s revenue for fiscal year 2022?'")
    st.write("- 'What are the major risk factors mentioned?'")
    st.write("- 'Who are Apple’s executive officers?' (non-financial but still in the 10-K)")

if __name__ == "__main__":
    main()
