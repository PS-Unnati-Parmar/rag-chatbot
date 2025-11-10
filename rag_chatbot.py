import os
import streamlit as st
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from bs4 import BeautifulSoup

# -------------------------------
# 1. Load environment variables
# -------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# -------------------------------
# 2. Helper functions
# -------------------------------

def crawl_website(url):
    """Fetch and extract visible text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract readable text
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:2000])  # Limit to avoid token overflow

    except Exception as e:
        return f"Error fetching website: {e}"

def generate_answer(context, query):
    """Generate an answer using Gemini based on context and user query."""
    prompt = f"""
You are a helpful web knowledge assistant.

Context from the website:
{context}

Question:
{query}

Answer in a concise and accurate way based on the above context.
If the answer is not found in context, clearly say "I couldn’t find that information on this site."
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🌐 Website Knowledge RAG Assistant (Gemini 2.5 Flash Lite)")
st.markdown("Enter a website URL to crawl and then ask questions based on its content.")

# Input section
url = st.text_input("🔗 Enter Website URL:")
fetch_button = st.button("📥 Fetch Website Content")

if fetch_button and url:
    with st.spinner("Fetching website content..."):
        context = crawl_website(url)
        st.session_state["context"] = context
    st.success("✅ Website content fetched successfully!")
    st.text_area("Website Content (First 2000 chars):", context[:2000], height=200)

# Question input
if "context" in st.session_state:
    query = st.text_input("❓ Ask a question about the website content:")
    if st.button("💬 Get Answer"):
        with st.spinner("Generating answer..."):
            answer = generate_answer(st.session_state["context"], query)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
