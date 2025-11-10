# 🌐 Website Knowledge RAG Assistant (Gemini 2.5 Flash Lite)

This project is a simple **Retrieval-Augmented Generation (RAG)** application built using **Streamlit** and **Google Gemini 2.5 Flash Lite**.  
It allows users to **enter any website URL**, automatically **fetch its content**, and then **ask natural language questions** based on that website’s text.

---

## 🚀 Features
✅ Uses **Gemini 2.5 Flash Lite** for fast and efficient LLM responses  
✅ Built with **Streamlit** for an easy-to-use web interface  
✅ **Crawls** and extracts text content from any user-provided URL  
✅ Secure **API key management** via `.env` file  
✅ Clean UI — no raw text shown to users  

---

## 🧠 How It Works
1. You enter a **website URL** in the Streamlit interface.  
2. The app fetches the **visible text** from that page using `requests` and `BeautifulSoup`.  
3. The text is used as **context** for Gemini.  
4. You can then ask **questions**, and Gemini answers based on the retrieved content.

---

## 🧩 Tech Stack
- **Python 3.9+**
- **Streamlit**
- **google-generativeai**
- **BeautifulSoup4**
- **python-dotenv**
- **Requests**

---

## 📦 Installation

1️⃣ Clone this repository
git clone https://github.com/PS-Unnati-Parmar/rag-chatbot.git



