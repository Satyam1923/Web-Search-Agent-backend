# Web Search Agent API

A **FastAPI** service that performs AI-assisted web searches, scrapes web content, and generates concise summaries.  
Built with **Playwright**, **LangChain**, **Google Gemini**, and **ChromaDB** for persistent storage and semantic search.

---

## 🚀 Features

- **AI Query Validation** – Filters out "human-only" tasks that can't be answered via search (e.g., *"Call my mom"*).
- **DuckDuckGo Search Integration** – Retrieves top search results.
- **Automated Web Scraping** – Uses Playwright to extract meaningful text.
- **Popup Handling** – Automatically dismisses cookie/privacy banners.
- **AI Summarization** – Uses Gemini (`gemini-2.5-pro`) to produce clear summaries.
- **ChromaDB Vector Store** – Stores queries, summaries, and sources persistently.
- **Semantic Similarity Matching** – Avoids redundant searches by comparing new queries with stored ones.

---

## 🛠 Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) – Web framework
- [Playwright](https://playwright.dev/python/) – Web automation & scraping
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [Google Generative AI](https://cloud.google.com/generative-ai) – Query validation & summarization
- [HuggingFace Embeddings](https://huggingface.co/) – Semantic vector embeddings
- [ChromaDB](https://www.trychroma.com/) – Vector database
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment management
- [NumPy](https://numpy.org/) – Cosine similarity calculations

---

## 📦 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/Web-Search-Agent-backend.git
cd web-search-api
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Install Playwright Browsers
```bash
playwright install
```

---

## 🔑 Environment Variables

Create a `.env` file with:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

If the API key is not set, you will be prompted for it at runtime.

---

## ▶️ Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will run at:  
**http://localhost:8000**

---

## 📡 API Endpoints

### **POST** `/search`
Performs a web search, scrapes the top results, summarizes content, and stores it.

**Request:**
```json
{
  "query": "latest AI trends 2025"
}
```

**Response:**
```json
{
  "query": "latest AI trends 2025",
  "normalized_query": "latest ai trends 2025",
  "summary": "AI in 2025 is focusing on multimodal models, edge AI adoption, and ethical AI frameworks...",
  "sources": [
    {
      "title": "Top AI Trends for 2025",
      "url": "https://example.com/ai-trends-2025"
    }
  ],
  "stored_queries": [
    "latest ai trends 2025",
    "best ai research papers"
  ]
}
```

---

## 📂 Project Structure

```
.
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── .env                   # API key storage
├── chroma_langchain_db/   # ChromaDB persistent storage
└── README.md              # Project documentation
```

---

## ⚙ How It Works

1. **Receive Query** – User submits a search term.
2. **Validate Query** – Gemini checks if it’s a human-only task.
3. **Check ChromaDB** – Searches for similar stored queries.
4. **Perform Search** – Uses DuckDuckGo to find top results.
5. **Scrape Pages** – Playwright retrieves relevant page content.
6. **Generate Summary** – Gemini produces a concise response.
7. **Store in ChromaDB** – Saves summary, query, and sources.

---

## 🛡 Error Handling

- **400** – Query requires human action (invalid for search).
- **500** – Scraping or summarization failures.
- Automatic retries for network and scraping errors.

---

## 📜 License

MIT License © 2025 Satyam Rathor
