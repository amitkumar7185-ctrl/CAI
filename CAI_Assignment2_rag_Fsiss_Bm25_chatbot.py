###############################################################################
# IMPORTS
###############################################################################
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
import pdfplumber
import sys
import glob
import os


print(sys.path)

###############################################################################
# DOWNLOAD NLTK DATASETS
###############################################################################
# Download tokenization and stopword datasets for text processing
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

###############################################################################
# DATA COLLECTION & PREPROCESSING
###############################################################################

def load_apple_filings(file_paths):
    """
    Load plain text Apple financial filings (.txt files).    
    Args:
        file_paths (list): List of file paths to text documents.
    Returns:
        list: List of text strings from the loaded files.
    """
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            texts.append(text)
    return texts

def load_apple_filings_pdf(file_paths):
    """
    Load PDF Apple financial filings and extract both text and tables.
    Args:
        file_paths (list): List of file paths to PDF documents.
    Returns:
        list: List of extracted text and tables as strings.
    """
    texts = []
    for path in file_paths:
        all_text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # Extract normal text
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    table_str = "\n".join(
                        [" | ".join([cell if cell is not None else "" for cell in row]) for row in table if row]
                    )
                    all_text += "\n[Extracted Table]\n" + table_str + "\n"
                    
        texts.append(all_text)
    return texts

def chunk_text(text, chunk_size=250):
    """
    Chunk large texts into smaller parts (~chunk_size words each).

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): Approximate maximum words per chunk.

    Returns:
        list: List of text chunks.
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
    End-to-end preprocessing: Load PDFs, chunk them, and prepare metadata.
    Args:
        file_paths (list): List of PDF paths.
    Returns:
        list: List of dicts with 'text' and 'source' (file name).
    """
    all_documents = []
    texts = load_apple_filings_pdf(file_paths)
    for i, text in enumerate(texts):
        file_name = os.path.basename(file_paths[i])
        chunks = chunk_text(text, chunk_size=250)
        for c in chunks:
            if len(c.strip()) > 0:
                all_documents.append({
                    "text": c.strip(),
                    "source": file_name
                })
    return all_documents

###############################################################################
# SETUP EMBEDDING MODEL, LANGUAGE MODEL & FAISS INDEX
###############################################################################

# Initialize the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)

# Initialize the language model (small open-source LM)
lm_model_name = "EleutherAI/gpt-neo-125M"  # distilGPT2 or you can use "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

# Global variables to store FAISS index and document texts
faiss_index = None
doc_texts = []  # list of chunk texts
doc_metadatas = []  # corresponding metadata (e.g., source)

def build_faiss_index(documents):
    """
    Build FAISS index for semantic search over the document chunks.
    Args:
        documents (list): Preprocessed document chunks.
    Returns:
        faiss.IndexFlatL2: FAISS index object.
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
# ADVANCED RETRIEVAL IMPLEMENTATION (WITH FAISS & Optional BM25)
###############################################################################

def retrieve_documents_faiss(query, top_k=3):
    """
    Retrieve top_k relevant document chunks from FAISS using the query.
    Returns retrieved texts along with their confidence scores.

    Args:
        query (str): User query.
        top_k (int): Number of top documents to retrieve.

    Returns:
        list: Retrieved (document_text, score) tuples.
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
    """
    Calculate BM25 score between query and a document.
    """
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
    """
    Retrieve top_k documents using BM25 when FAISS fails.

    Args:
        query (str): User query.
        documents (list): Document corpus.
        top_k (int): Number of documents to retrieve.

    Returns:
        list: Retrieved (document_text, score) tuples.
    """
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
# GUARDRAILS & RESPONSE GENERATION
###############################################################################

def guardrail_input(user_input):
    """
    Validate input query to ensure it's safe and appropriate.
    Returns:
        tuple: (bool for valid input, error message if any)
    """
    if not user_input or len(user_input.strip()) == 0:
        return False, "Query cannot be empty."
    if len(user_input) > 1000:
        return False, "Query too long."
    return True, ""

def guardrail_output(response_text):
    """
    Add guardrails to model output if needed.
    """
    if "I am not sure" in response_text:
        return "[Guardrail] The model seems uncertain. Please verify the response."
    return response_text

def generate_response(query, retrieved_docs, conversation_history):
    """
    Generate response from the language model using retrieved context.
    Args:
        query (str): User query.
        retrieved_docs (list): Retrieved chunks.
        conversation_history (list): Past conversation turns.
    Returns:
        str: Generated response text.
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
            max_length=1500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return output_text.strip()

###############################################################################
# UI DEVELOPMENT (STREAMLIT)
###############################################################################

def get_all_txt_files(folder_path):
    """
    Retrieve all .pdf file paths from the given folder.
    """
    return glob.glob(os.path.join(folder_path, "*.pdf"))
def load_preprocess_all_file():   
    """
    Load and preprocess all PDFs in 'Data' folder and build the FAISS index.
    """ 
    # Define your folder path
    folder_path = r"Data"
    # Get all .txt files from the folder
    file_paths = get_all_txt_files(folder_path)  
    documents = preprocess_apple_data(file_paths)
    print(f"Total chunks created: {len(documents)}")
    build_faiss_index(documents)
    st.session_state.documents = documents
    st.success("Indexes built successfully.")


def main():
    st.title("Financial RAG Chatbot with FAISS")
    load_preprocess_all_file()
    # Initialize session state for conversation and document storage
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = None    

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

    # Display conversation history with confidence scores
    st.subheader("2. Conversation History")
    for turn in st.session_state.conversation_history:
        st.write(f"**User:** {turn['user']}")
        for i, (doc, score) in enumerate(turn['retrieved_docs']):
            st.write(f"**Retrieved Doc {i+1} Confidence Score:** {score:.4f}")
        st.write(f"**Assistant:** {turn['assistant']}")

    # # Display retrieved documents with confidence scores
    # st.subheader("3.Retrieved Documents with Confidence Scores:")
    # for i, (doc, score) in enumerate(retrieved_docs_with_scores):
    #     st.write(f"**Document {i+1}:**")
    #     st.write(f"**Confidence Score:** {score:.4f}")
    #     st.write(doc)  


    st.subheader("3. Testing & Validation")
    st.write("Try queries like:")
    st.write("- 'What is Apple’s revenue for fiscal year 2022?'")
    st.write("- 'What are the major risk factors mentioned?'")
    st.write("- 'Who are Apple’s executive officers?' (non-financial but still in the 10-K)")

if __name__ == "__main__":
    main()
