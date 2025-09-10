from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import getpass
import asyncio
import string
from collections import Counter
import json
from playwright.async_api import async_playwright, TimeoutError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from numpy import dot
import re
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

def safe_json_extract(raw_text: str):
    """Try to extract JSON list from model output safely."""
    try:
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Safe JSON extraction failed: {e}")
    return []


@app.get("/")
def hello():
    return {
        "message":"Hello world"
    }


# ------------------ Search Endpoint -----------------
@app.post("/search", response_model=SearchResponse)
async def perform_search(search_query: SearchQuery):
    """Perform a web search and return a summary with sources + ranked entities."""
    user_query = search_query.query.strip()
    normalized_query = normalize_query(user_query)
    logger.info(f"Normalized query: {normalized_query}")

    # --- Query validation ---
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

    # --- Semantic similarity check ---
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
            logger.info(f"Debug cosine similarity: {debug_similarity}")
            if debug_similarity > 0.8:
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
                logger.info("No sufficiently similar query found, proceeding to scrape.")
        else:
            logger.info("No matching queries found in ChromaDB.")
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")

    # --- Web search with DuckDuckGo ---
    logger.info("Searching DuckDuckGo...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        search_url = f"https://duckduckgo.com/?q={user_query.replace(' ', '+')}"

        retries = 3
        elements = None
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
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    raise HTTPException(status_code=500, detail="Max retries reached.")
                await page.wait_for_timeout(2000)

        # --- Collect top results ---
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

        if not top_results:
            raise HTTPException(status_code=500, detail="No valid search results found.")

        # --- Scrape pages + extract entities ---
        all_scraped_texts = []
        sources = []
        extracted_entities = []
        scraped_snippets = "\n\n".join([text[:2000] for text in all_scraped_texts])
        with open("scraped_pages.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {user_query}\n")
            f.write(f"Normalized Query: {normalized_query}\n\n")

        logger.info("Top 5 Search Results:")
        for i, result in enumerate(top_results, start=1):
            logger.info(f"[{i}] {result['title']}\nURL: {result['url']}")
            text = await scrape_page(page, result["url"])
            if text:
                all_scraped_texts.append(text)
                sources.append({"url": result["url"], "title": result["title"]})

                # Save scraped raw content
                with open("scraped_pages.txt", "a", encoding="utf-8") as f:
                    f.write(f"--- Page {i}: {result['title']} ---\n")
                    f.write(f"URL: {result['url']}\n\n")
                    f.write(text)
                    f.write("\n\n" + "="*100 + "\n\n")

                # --- Extract entities with Gemini ---
                extract_prompt = (
                    f"Extract the key entities relevant to the user query.\n\n"
                    f"User Query: {user_query}\n\n"
                    f"Page Title: {result['title']}\n"
                    f"Page Content:\n{text[:3000]}\n\n"
                    "Return ONLY a JSON list of entities like this: [\"Entity1\", \"Entity2\", ...]"
                )
                try:
                    extraction = model.invoke(extract_prompt)
                    if hasattr(extraction, "content"):
                        entities = safe_json_extract(extraction.content)
                        if entities:
                            extracted_entities.append({
                                "source_rank": i,  # lower rank = higher priority
                                "url": result["url"],
                                "title": result["title"],
                                "entities": entities
                            })
                except Exception as e:
                    logger.warning(f"Entity extraction failed for {result['url']}: {e}")

        await context.close()
        await browser.close()

        if not all_scraped_texts:
            raise HTTPException(status_code=500, detail="No valid content scraped from any page.")

        # --- Merge and rank entities (majority voting + weighted by source rank) ---
        from collections import defaultdict

        def rank_entities(extracted_entities):
            scores = defaultdict(float)
            counts = defaultdict(int)

            for site in extracted_entities:
                source_weight = 1.0 / site["source_rank"]  # higher rank → higher weight
                for pos, entity in enumerate(site["entities"], start=1):
                    scores[entity] += source_weight / pos
                    counts[entity] += 1

            ranked = [
                (entity, round(score, 4), counts[entity])
                for entity, score in scores.items()
            ]
            ranked_sorted = sorted(ranked, key=lambda x: (-x[1], -x[2]))
            return ranked_sorted

        ranked_entities = rank_entities(extracted_entities)
        logger.info(f"Ranked entities: {ranked_entities}")

        # --- Save ranked entities ---
        with open("ranked_entities.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {user_query}\n")
            f.write(f"Normalized Query: {normalized_query}\n\n")
            f.write("Ranked Entities (entity -> score | count):\n\n")
            for idx, (entity, score, count) in enumerate(ranked_entities, start=1):
                f.write(f"{idx}. {entity} -> {score} | {count}\n")

        # --- Generate summary ---
        logger.info("Generating summary with Gemini...")
        summary_prompt = (
    f"User query: {user_query}\n\n"
    "Ranked entities with scores and frequency:\n"
    f"{json.dumps(ranked_entities, indent=2)}\n\n"
    "Scraped content snippets from different sources:\n"
    f"{scraped_snippets}\n\n"
    "Write a comprehensive, descriptive answer:\n"
    "- Clearly mention the top-ranked entities.\n"
    "- Provide a 3–5 sentence description of each, using details from the scraped text.\n"
    "- If sources disagree, highlight the differences.\n"
    "- Use an informative and engaging tone suitable for readers."
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

        # --- Store in ChromaDB ---
        try:
            logger.info(f"Storing query: {normalized_query}")
            pre_store_embedding = embeddings.embed_query(normalized_query)
            vector_store.add_texts(
                [summary_text],
                metadatas=[{
                    "query": normalized_query,
                    "summary": summary_text,
                    "sources": json.dumps(sources),
                    "ranked_entities": json.dumps(ranked_entities)
                }]
            )
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
