from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import getpass
import asyncio
import string
import json
from playwright.async_api import async_playwright, TimeoutError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from numpy import dot
from numpy.linalg import norm
from typing import List, Dict
import logging

# ------------------ Setup Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Web Search API", description="API for performing web searches and generating summaries.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Pydantic Models ------------------
class SearchQuery(BaseModel):
    query: str

class Source(BaseModel):
    title: str
    url: str

class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    summary: str
    sources: List[Source]
    stored_queries: List[str]

# ------------------ Environment Setup ------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    google_api_key = getpass.getpass("Enter your Google API key: ")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment or entered manually.")

# ------------------ Utility Functions ------------------
def normalize_query(query: str) -> str:
    """Normalize query for consistent comparison without synonyms."""
    query = query.lower().strip()
    query = query.translate(str.maketrans('', '', string.punctuation))
    return query.strip()

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# ------------------ Initialize Embeddings & ChromaDB ------------------
persist_directory = "./chroma_langchain_db"
client = chromadb.PersistentClient(path=persist_directory)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="web_search")
vector_store = Chroma(
    collection_name="web_search",
    embedding_function=embeddings,
    persist_directory=persist_directory,
    client=client
)

def list_stored_queries() -> List[str]:
    """List all stored queries in ChromaDB."""
    try:
        results = vector_store._collection.get(include=["metadatas"])
        queries = [meta["query"] for meta in results["metadatas"]]
        logger.info(f"Stored queries in ChromaDB: {queries}")
        return queries
    except Exception as e:
        logger.error(f"Error listing stored queries: {e}")
        return []

logger.info(f"Initial number of documents in ChromaDB: {collection.count()}")
list_stored_queries()

# ------------------ Query Validation Prompt ------------------
validation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a binary classifier. Your job is to determine whether a user's input is a task that requires **human action** and cannot be performed by a search engine.\n\n"
        "Tasks that require a human include physical actions, reminders, personal errands, or actions that need personal interaction or execution. These cannot be answered by searching online.\n\n"
        "Examples of such tasks:\n"
        "- Walk my dog\n"
        "- Call my mom\n"
        "- Add milk to my shopping list\n"
        "- Send an email to Alex\n"
        "- Schedule a meeting at 5 PM\n\n"
        "Examples of things that are **not** human-only tasks (these can be answered by a search engine):\n"
        "- How to walk a dog properly?\n"
        "- Best shopping list apps\n"
        "- How to schedule a Google Calendar event?\n"
        "- What to say in an email to a professor?\n"
        "- What’s the weather tomorrow in Bangalore?\n\n"
        "Reply ONLY with:\n"
        "'yes' → if the input is a human-only task\n"
        "'no' → if it is **not** a human-only task\n"
        "No other text or explanation."
    )),
    HumanMessage(content="{query}")
])

# ------------------ Gemini Model ------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=google_api_key , temperature=0.1)

# ------------------ Popup Handling ------------------
async def handle_popups(page):
    """Attempt to dismiss common pop-ups like cookie consent banners."""
    popup_selectors = [
        "button:text('Accept')",
        "button:text('Accept All')",
        "button:text(' Agree')",
        "button:text('OK')",
        "button:text('Close')",
        "button:text('Decline')",
        "a:text('Accept')",
        "a:text('Close')",
        "[id*='cookie'] button",
        "[class*='cookie'] button",
        "[id*='privacy'] button",
        "[class*='privacy'] button",
    ]
    for selector in popup_selectors:
        try:
            elements = await page.locator(selector).element_handles()
            for element in elements:
                await element.click()
                logger.info(f"Dismissed popup with selector: {selector}")
                await page.wait_for_timeout(1000)
        except Exception as e:
            logger.warning(f"Could not dismiss popup with selector {selector}: {e}")

# ------------------ Scrape Page ------------------
async def scrape_page(page, url: str, retries: int = 3, timeout: int = 80000) -> str:
    """Attempt to scrape a page with retries."""
    for attempt in range(retries):
        try:
            logger.info(f"Attempting to scrape {url} (attempt {attempt + 1}/{retries})")
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            await handle_popups(page)
            text = await page.locator("body").inner_text(timeout=15000)
            if text and len(text.strip()) > 100:
                logger.info(f"Successfully scraped {url}")
                return text.strip()
            logger.warning(f"Insufficient content scraped from {url}.")
        except TimeoutError:
            logger.warning(f"Timeout (attempt {attempt + 1}/{retries}) while scraping {url}.")
        except Exception as e:
            logger.warning(f"Failed to scrape {url} (attempt {attempt + 1}/{retries}): {e}")
    return None

# ------------------ Search Endpoint ------------------
@app.post("/search", response_model=SearchResponse)
async def perform_search(search_query: SearchQuery):
    """Perform a web search and return a summary with sources."""
    user_query = search_query.query.strip()
    normalized_query = normalize_query(user_query)
    logger.info(f"Normalized query: {normalized_query}")

    # Query validation
    try:
        chain = validation_prompt | model
        response = chain.invoke({"query": normalized_query})
        if not response or not hasattr(response, "content"):
            raise HTTPException(status_code=500, detail="No validation response received.")
        verdict = response.content.strip().lower()
        logger.info(f"Is it a human-only task? → {verdict}")
        if verdict == "yes":
            raise HTTPException(status_code=400, detail="This is not a valid query.")
        elif verdict not in {"yes", "no"}:
            logger.error(f"Unexpected verdict from validation model: {verdict}")
            raise HTTPException(status_code=500, detail="Invalid response from validation model.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during query validation: {e}")
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")

    # Semantic similarity check
    try:
        results = vector_store.similarity_search_with_score(normalized_query, k=1)
        if results:
            doc, score = results[0]
            logger.info(f"Similarity score: {score}")
            logger.info(f"Matched stored query: {doc.metadata['query']}")
            stored_query = doc.metadata['query']
            stored_embedding = embeddings.embed_query(stored_query)
            current_embedding = embeddings.embed_query(normalized_query)
            debug_similarity = cosine_similarity(stored_embedding, current_embedding)
            logger.info(f"Debug cosine similarity between '{normalized_query}' and stored '{stored_query}': {debug_similarity}")
            if debug_similarity > 0.8:  # Use cosine similarity threshold
                logger.info("Query already exists in ChromaDB.")
                summary_text = doc.metadata.get("summary", "No summary available.")
                sources_json = doc.metadata.get("sources", "[]")
                sources = json.loads(sources_json) if sources_json else []
                return SearchResponse(
                    query=user_query,
                    normalized_query=normalized_query,
                    summary=summary_text,
                    sources=[Source(**source) for source in sources],
                    stored_queries=list_stored_queries()
                )
            else:
                logger.info(f"No sufficiently similar query found (cosine similarity: {debug_similarity}). Proceeding to scrape.")
        else:
            logger.info("No matching queries found in ChromaDB.")
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")

    logger.info("Searching DuckDuckGo...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        search_url = f"https://duckduckgo.com/?q={user_query.replace(' ', '+')}"
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Attempting to load search page (attempt {attempt + 1}/{retries})")
                await page.goto(search_url, wait_until="networkidle", timeout=80000)
                await handle_popups(page)
                selectors = [
                    "a[data-testid='result-title-a']",
                    ".result__a",
                    "h2 > a",
                ]
                elements = None
                for selector in selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=10000)
                        elements = await page.locator(selector).element_handles()
                        if elements:
                            logger.info(f"Using selector: {selector}")
                            break
                    except TimeoutError:
                        logger.warning(f"Selector {selector} not found.")
                if not elements:
                    raise HTTPException(status_code=500, detail="No search results found with any selector.")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed to load search page: {e}")
                if attempt == retries - 1:
                    raise HTTPException(status_code=500, detail="Max retries reached. Failed to load search results page.")
                await page.wait_for_timeout(2000)

        top_results = []
        for elem in elements[:10]:
            href = await elem.get_attribute("href")
            inner_text = await elem.inner_text()
            if href and href.startswith("http") and inner_text:
                top_results.append({
                    "title": inner_text.strip(),
                    "url": href.strip()
                })
            if len(top_results) == 5:
                break

        all_scraped_texts = []
        sources = []
        if not top_results:
            raise HTTPException(status_code=500, detail="No valid search results found.")

        logger.info("Top 5 Search Results:")
        for i, result in enumerate(top_results, start=1):
            logger.info(f"[{i}] {result['title']}\nURL: {result['url']}")
            text = await scrape_page(page, result["url"])
            if text:
                all_scraped_texts.append(text)
                sources.append({"url": result["url"], "title": result["title"]})

        await context.close()
        browser.close()

        if not all_scraped_texts:
            raise HTTPException(status_code=500, detail="No valid content scraped from any page.")

        logger.info("Generating summary with Gemini...")
        max_content_length = 5000
        truncated_texts = [text[:max_content_length] for text in all_scraped_texts]
        summary_prompt = (
            "Summarize the following content into a concise and informative response "
            "based on the user's query:\n\n"
            f"User query: {user_query}\n\n"
            f"Scraped Content:\n\n" + "\n\n---\n\n".join(truncated_texts)
        )

        try:
            summary = model.invoke(summary_prompt)
            if hasattr(summary, "content"):
                summary_text = summary.content.strip()
            else:
                raise HTTPException(status_code=500, detail="Failed to generate summary.")
        except Exception as e:
            logger.error(f"Error during summary generation: {e}")
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

        # Store in ChromaDB
        try:
            logger.info(f"Storing query: {normalized_query}")
            pre_store_embedding = embeddings.embed_query(normalized_query)
            vector_store.add_texts(
                [summary_text],
                metadatas=[{
                    "query": normalized_query,
                    "summary": summary_text,
                    "sources": json.dumps(sources)
                }]
            )
            results = vector_store.similarity_search_with_score(normalized_query, k=1)
            if results:
                doc, score = results[0]
                logger.info(f"Stored query: {doc.metadata['query']}, Similarity score: {score}")
                post_store_embedding = embeddings.embed_query(doc.metadata['query'])
                debug_similarity = cosine_similarity(pre_store_embedding, post_store_embedding)
                logger.info(f"Debug cosine similarity between pre-store and post-store '{normalized_query}': {debug_similarity}")
            else:
                logger.error("Document was not stored in ChromaDB.")
            logger.info(f"Total documents in ChromaDB: {collection.count()}")
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")

        return SearchResponse(
            query=user_query,
            normalized_query=normalized_query,
            summary=summary_text,
            sources=[Source(**source) for source in sources],
            stored_queries=list_stored_queries()
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)