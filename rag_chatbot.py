import os
import streamlit as st
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = get_embedder()

def fast_crawl(url, timeout=7):
    """Fetch only visible relevant text from main tags quickly."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]): tag.decompose()
        main = soup.find('main')
        article = soup.find('article')
        div = soup.find('div')
        text = ""
        if main: text = main.get_text(separator="\n")
        elif article: text = article.get_text(separator="\n")
        elif div: text = div.get_text(separator="\n")
        else: text = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    except Exception as e:
        return f"Error: {e}"

def chunk(text, n=256, overlap=32):
    """Quick chunker."""
    # Split text and filter out any empty chunk
    raw_chunks = [text[i:i+n] for i in range(0, len(text), n-overlap)]
    return [c for c in raw_chunks if c.strip()]

def make_index(chunks, embedder):
    if not chunks:
        raise ValueError("No valid (non-empty) chunks to index.")
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    if len(embeddings.shape) != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embeddings array has invalid shape: {embeddings.shape}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_top_chunks(question, index, chunks, embedder, k=2):
    q_emb = embedder.encode([question]).astype('float32')
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]

def get_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely based on context."
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as ex:
        return f"Error: {ex}"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🌐 Website RAG Chatbot")

if "msgs" not in st.session_state: st.session_state.msgs = []

with st.sidebar:
    url = st.text_input("Enter Website URL:", placeholder="https://example.com")
    if st.button("🔄 Fetch & Index", use_container_width=True):
        if url:
            with st.spinner("Fetching website..."):
                page = fast_crawl(url)
                if page.startswith("Error:"):
                    st.error(page)
                else:
                    chunks = chunk(page, n=256, overlap=32)
                    chunks = [c for c in chunks if c.strip()]  # Extra filter, just in case
                    if not chunks:
                        st.error("No usable content found on this website.")
                    else:
                        try:
                            index = make_index(chunks, embedding_model)
                            st.session_state.index = index
                            st.session_state.chunks = chunks
                            st.session_state.msgs = []
                            st.session_state.url = url
                        except Exception as ex:
                            st.error(f"Indexing failed: {ex}")
        else:
            st.warning("Enter a URL first.")

if "index" in st.session_state:
    st.subheader("💬 Chat")
    for msg in st.session_state.msgs:
        who, text = msg["role"], msg["content"]
        with st.chat_message(who): st.write(text)
    in_col, btn_col = st.columns([5,1])
    with in_col:
        q = st.text_input("Ask a question:", "", key="chat_input")
    with btn_col:
        if st.button("Send", use_container_width=True) and q.strip():
            st.session_state.msgs.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            with st.spinner("Thinking..."):
                context = "\n\n".join(get_top_chunks(q, st.session_state.index, st.session_state.chunks, embedding_model))
                ans = get_answer(context, q)
            st.session_state.msgs.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.write(ans)
            st.rerun()
else:
    st.info("👉 Fetch a website in the sidebar first to use the chat.")
