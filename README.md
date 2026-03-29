# 🛒 Flipkart Product Recommender — Complete Project Guide

> **An end-to-end RAG-based (Retrieval-Augmented Generation) chatbot that recommends Flipkart products using real customer reviews, powered by LangChain, Groq LLaMA 3.1, AstraDB, and deployed on Kubernetes with Prometheus + Grafana monitoring.**

---

## 📌 Table of Contents

1. [What This Project Actually Does](#-what-this-project-actually-does)
2. [How It Works — The Big Picture](#-how-it-works--the-big-picture)
3. [Tech Stack Breakdown](#-tech-stack-breakdown)
4. [Project Structure — File by File](#-project-structure--file-by-file)
5. [End-to-End Pipeline Deep Dive](#-end-to-end-pipeline-deep-dive)
   - [Step 1: The Dataset](#step-1-the-dataset)
   - [Step 2: Data Conversion (CSV → LangChain Documents)](#step-2-data-conversion-csv--langchain-documents)
   - [Step 3: Embedding & Vector Storage (AstraDB)](#step-3-embedding--vector-storage-astradb)
   - [Step 4: RAG Chain (The Brain)](#step-4-rag-chain-the-brain)
   - [Step 5: Flask Web App (The Interface)](#step-5-flask-web-app-the-interface)
   - [Step 6: Frontend (Chat UI)](#step-6-frontend-chat-ui)
6. [Configuration & Environment Variables](#-configuration--environment-variables)
7. [Utility Modules (Logging & Exception Handling)](#-utility-modules-logging--exception-handling)
8. [How to Run This Project Locally](#-how-to-run-this-project-locally)
9. [Docker Containerization](#-docker-containerization)
10. [Kubernetes Deployment](#-kubernetes-deployment)
11. [Monitoring with Prometheus & Grafana](#-monitoring-with-prometheus--grafana)
12. [How the Conversation Flow Works (Start to Finish)](#-how-the-conversation-flow-works-start-to-finish)

---

## 🎯 What This Project Actually Does

Imagine you're shopping on Flipkart and you want to know: *"Which Bluetooth headset has the best bass?"* or *"Is the BoAt Rockerz 235v2 worth buying?"*

This project builds an **AI chatbot** that answers exactly those kinds of questions — but instead of making things up, it **searches through real customer reviews** from Flipkart, finds the most relevant ones, and then uses an LLM (Large Language Model) to generate a helpful, contextual answer based on what actual customers have said.

**This is NOT a traditional recommendation system** (no collaborative filtering, no matrix factorization). It's a **RAG (Retrieval-Augmented Generation)** system — meaning the AI retrieves relevant documents first, then generates an answer grounded in those documents. This makes the responses factual and rooted in real customer opinions.

---

## 🔄 How It Works — The Big Picture

Here's the complete flow from data to answer:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                            │
│                                                                        │
│  CSV Dataset (450 reviews, 9 products)                                 │
│       │                                                                │
│       ▼                                                                │
│  DataConverter: Reads CSV → Creates LangChain Document objects         │
│  (each doc = one review as page_content + product name as metadata)    │
│       │                                                                │
│       ▼                                                                │
│  DataIngestor: Embeds documents using HuggingFace BGE model            │
│  → Stores vectors in AstraDB (cloud Cassandra vector database)         │
│       │                                                                │
│       ▼                                                                │
│  RAGChainBuilder: On every user query —                                │
│    1. Rewrites the query using chat history (context-aware)            │
│    2. Retrieves top 3 most similar reviews from AstraDB                │
│    3. Feeds reviews + query to Groq LLaMA 3.1 8B                      │
│    4. LLM generates a concise, contextual answer                       │
│    5. Updates conversation history for follow-up questions             │
│       │                                                                │
│       ▼                                                                │
│  Flask App: Serves the chat UI + handles API requests                  │
│  → Exposes /metrics endpoint for Prometheus monitoring                 │
│       │                                                                │
│       ▼                                                                │
│  Docker → Kubernetes → Prometheus → Grafana (production deployment)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧱 Tech Stack Breakdown

| Component | Technology | Why It's Used |
|-----------|------------|---------------|
| **LLM (Answer Generation)** | Groq + LLaMA 3.1 8B Instant | Groq provides ultra-fast inference for open-source LLMs. LLaMA 3.1 8B is powerful yet lightweight |
| **Embedding Model** | HuggingFace `BAAI/bge-base-en-v1.5` | Best-in-class open-source embedding model for semantic search. Converts text → 768-dim vectors |
| **Vector Database** | DataStax AstraDB | Cloud-hosted Cassandra with native vector search. No infrastructure management needed |
| **Orchestration Framework** | LangChain | Chains together retrieval, prompting, and LLM calls. Handles conversation history automatically |
| **Web Framework** | Flask | Lightweight Python web framework. Serves the chat UI and API |
| **Frontend** | HTML + CSS + jQuery + Bootstrap | Simple chat interface with dark theme. jQuery handles async message sending |
| **Monitoring** | Prometheus + Grafana | Prometheus scrapes app metrics, Grafana visualizes them in dashboards |
| **Containerization** | Docker | Packages the entire app into a portable container |
| **Orchestration** | Kubernetes | Deploys and manages containers at scale with auto-healing and load balancing |
| **Configuration** | python-dotenv | Loads sensitive API keys from `.env` file so they're never hardcoded |

---

## 📁 Project Structure — File by File

```
flipkart_product_recommender/
│
├── app.py                          # 🚀 Main entry point — Flask app with routes
├── setup.py                        # 📦 Package installer — makes project pip-installable
├── requirements.txt                # 📋 All Python dependencies with pinned versions
├── DockerFile                      # 🐳 Docker build instructions
├── .gitignore                      # 🚫 Files excluded from Git
├── .env                            # 🔐 API keys (not committed — you create this)
│
├── data/
│   └── flipkart_product_review.csv # 📊 Dataset: 450 reviews across 9 headset products
│
├── flipkart/                       # 🧠 Core ML/RAG pipeline package
│   ├── __init__.py                 # Makes this directory a Python package
│   ├── config.py                   # ⚙️ Centralized configuration (API keys, model names)
│   ├── data_converter.py           # 🔄 Converts CSV rows into LangChain Document objects
│   ├── data_ingestion.py           # 📥 Embeds documents and stores them in AstraDB
│   └── rag_chain.py                # 🔗 Builds the full RAG pipeline with chat history
│
├── utils/                          # 🛠 Utility modules
│   ├── __init__.py                 # Makes this directory a Python package
│   ├── logger.py                   # 📝 Logging configuration (date-based log files)
│   └── custom_exception.py         # ⚠️ Custom exception class with detailed error info
│
├── templates/
│   └── index.html                  # 🖥 Chat UI template (Jinja2 + Bootstrap + jQuery)
│
├── static/
│   └── style.css                   # 🎨 Dark-themed CSS for the chat interface
│
├── flask-deployment.yaml           # ☸️ Kubernetes Deployment + Service for Flask app
│
├── prometheus/
│   ├── prometheus-configmap.yaml   # ☸️ Prometheus scraping configuration
│   └── prometheus-deployment.yaml  # ☸️ Kubernetes Deployment + Service for Prometheus
│
└── grafana/
    └── grafana-deployment.yaml     # ☸️ Kubernetes Deployment + Service for Grafana
```

---

## 🔬 End-to-End Pipeline Deep Dive

### Step 1: The Dataset

**File:** `data/flipkart_product_review.csv`

This is where everything starts. The dataset contains **450 real customer reviews** for **9 different headset/earphone products** from Flipkart.

**CSV columns:**

| Column | Description | Example |
|--------|-------------|---------|
| `product_id` | Flipkart's unique product identifier | `ACCFZGAQJGYCYDCM` |
| `product_title` | Full product name | `BoAt Rockerz 235v2 with ASAP charging Version 5.0 Bluetooth Headset` |
| `rating` | Customer rating (1-5 stars) | `4` |
| `summary` | One-line review summary | `Terrific purchase` |
| `review` | Full detailed review text | `"1-more flexible 2-bass is very high 3-sound clarity is good..."` |

**The 9 products in the dataset:**
1. BoAt Rockerz 235v2 with ASAP charging Bluetooth Headset
2. realme Buds Wireless Bluetooth Headset
3. OnePlus Bullets Wireless Z Bluetooth Headset
4. realme Buds 2 Wired Headset
5. OnePlus Bullets Wireless Z Bass Edition Bluetooth Headset
6. realme Buds Q Bluetooth Headset
7. U&I Titanic Series - Low Price Bluetooth Neckband Headset
8. BoAt Airdopes 131 Bluetooth Headset
9. BoAt BassHeads 100 Wired Headset

Each product has about 50 reviews, giving the chatbot a rich set of real opinions to draw from.

---

### Step 2: Data Conversion (CSV → LangChain Documents)

**File:** `flipkart/data_converter.py`

```python
import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[["product_title", "review"]]

        docs = [
            Document(
                page_content=row['review'],
                metadata={"product_name": row["product_title"]}
            )
            for _, row in df.iterrows()
        ]

        return docs
```

**What this does, line by line:**

1. **Reads the CSV** using pandas, but only keeps two columns: `product_title` and `review`. The rating and summary aren't used — the RAG system works purely from the review text.

2. **Creates LangChain `Document` objects** — this is a key concept. LangChain's `Document` class has two parts:
   - `page_content`: The text that will be embedded and searched. Here, that's the **review text** itself.
   - `metadata`: Extra info attached to each document. Here, that's the **product name**, so when the system retrieves a review, it also knows which product it's about.

3. **Returns a list of 450 Document objects**, one per review.

**Why only `review` and `product_title`?** The review text is what contains the actual opinions (bass quality, battery life, comfort, etc.). The product title is stored as metadata so the LLM knows which product the review belongs to. The rating and summary are redundant since the review text already captures the customer's sentiment.

---

### Step 3: Embedding & Vector Storage (AstraDB)

**File:** `flipkart/data_ingestion.py`

```python
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config

class DataIngestor:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, load_existing=True):
        if load_existing == True:
            return self.vstore

        docs = DataConverter("data/flipkart_product_review.csv").convert()
        self.vstore.add_documents(docs)
        return self.vstore
```

**What this does, step by step:**

1. **Initializes the embedding model** — `BAAI/bge-base-en-v1.5` from HuggingFace. This model converts any text into a 768-dimensional vector. It's one of the top-performing open-source embedding models. The model runs locally (downloaded from HuggingFace Hub the first time).

2. **Connects to AstraDB** — DataStax AstraDB is a cloud-hosted Cassandra database with built-in vector search capabilities. It creates (or connects to) a collection called `"flipkart_database"`. The connection requires three credentials: API endpoint, authentication token, and keyspace name.

3. **The `ingest()` method has two modes:**
   - `load_existing=True` (default): Just returns the vector store connection. This is used during normal app startup — the data is already in AstraDB, so no need to re-embed and re-upload.
   - `load_existing=False`: Runs the full pipeline — reads CSV → converts to Documents → generates embeddings → uploads vectors to AstraDB. You only need to run this **once** to populate the database.

4. **The standalone `__main__` block** at the bottom lets you run this file directly (`python -m flipkart.data_ingestion`) to do the initial data ingestion.

**What happens during `add_documents()`:**
- Each review text is sent through the embedding model
- The model outputs a 768-dimensional vector representing the semantic meaning of that review
- The vector + the review text + the metadata (product name) are all stored in AstraDB
- AstraDB indexes these vectors for fast similarity search

---

### Step 4: RAG Chain (The Brain)

**File:** `flipkart/rag_chain.py`

This is the most important file — it's where the AI reasoning happens.

```python
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
```

**Breaking down the entire chain component by component:**

#### 4a. The LLM — Groq + LLaMA 3.1 8B

```python
self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
```

- **Groq** is an inference platform that runs LLMs on custom LPU (Language Processing Unit) hardware — it's extremely fast.
- **LLaMA 3.1 8B Instant** is Meta's open-source model. The "8B" means 8 billion parameters. "Instant" is Groq's optimized variant for low-latency responses.
- **`temperature=0.5`**: Controls randomness. 0 = deterministic, 1 = very creative. 0.5 is a sweet spot — gives some variety while staying grounded.

#### 4b. The Retriever

```python
retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
```

- Converts AstraDB vector store into a LangChain retriever
- **`k=3`**: For every query, it returns the **3 most semantically similar** reviews from the database
- Similarity is measured by cosine distance between the query embedding and stored review embeddings

#### 4c. Context-Aware Prompt (Query Rewriting)

```python
context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and user question, rewrite it as a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
```

**Why this exists:** Imagine this conversation:
- User: *"Tell me about BoAt Rockerz 235v2"*
- Bot: *"It has great bass and 6-8 hour battery life..."*
- User: *"What about the sound quality?"*

That second question — *"What about the sound quality?"* — is meaningless without context. Sound quality of **what**? This prompt tells the LLM: "Look at the chat history, look at this new question, and rewrite it as a complete standalone question." So it becomes: *"What is the sound quality of the BoAt Rockerz 235v2?"*

This rewritten question is then used to search the vector database, getting much better search results.

#### 4d. History-Aware Retriever

```python
history_aware_retriever = create_history_aware_retriever(
    self.model, retriever, context_prompt
)
```

This chains together: **LLM rewrites query** → **rewritten query searches vector DB** → **returns relevant documents**.

#### 4e. QA Prompt (Answer Generation)

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                  Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
```

- The `{context}` placeholder gets filled with the 3 retrieved reviews
- The `{input}` placeholder gets the user's question
- **"Stick to context"** is crucial — this instruction tells the LLM to only use information from the retrieved reviews, preventing hallucination
- The chat history placeholder allows the LLM to maintain conversational flow

#### 4f. Stuff Documents Chain

```python
question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
```

"Stuff" is LangChain's simplest document strategy — it literally stuffs all retrieved documents into the prompt. Since we only retrieve 3 reviews (each a few sentences), this fits easily within the context window. Other strategies like Map-Reduce or Refine exist for larger document sets, but aren't needed here.

#### 4g. Full RAG Chain

```python
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

This connects: **retrieval** (find relevant reviews) → **generation** (answer the question using those reviews).

#### 4h. Adding Message History

```python
return RunnableWithMessageHistory(
    rag_chain,
    self._get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```

- Wraps the entire chain with automatic conversation history management
- **`_get_history()`**: Creates a new `ChatMessageHistory` object per session ID, or retrieves an existing one. This is stored in-memory (`self.history_store` dict)
- Every time the chain runs, it automatically saves the user's input and the bot's response to history
- Next time the chain runs with the same session ID, it passes the full conversation history to both prompts

**The complete flow when a user asks a question:**

```
User: "Which headset has the best battery?"
           │
           ▼
   [1] Context Prompt: Rewrites question using chat history
           │
           ▼  
   [2] Rewritten query → Embedding → AstraDB similarity search
           │
           ▼
   [3] Top 3 matching reviews retrieved
           │
           ▼
   [4] QA Prompt: Reviews + Question + History → Groq LLaMA 3.1
           │
           ▼
   [5] LLM generates answer grounded in retrieved reviews
           │
           ▼
   [6] Answer + conversation saved to history
           │
           ▼
   [7] Answer returned to user
```

---

### Step 5: Flask Web App (The Interface)

**File:** `app.py`

```python
from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

from dotenv import load_dotenv
load_dotenv()

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")

def create_app():
    app = Flask(__name__)

    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        user_input = request.form["msg"]

        response = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user-session"}}
        )["answer"]

        return response

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
```

**What each part does:**

1. **`create_app()` — App Factory Pattern:**
   - This is Flask's recommended pattern. Instead of creating a global `app`, a function creates and configures it. This makes testing and multiple instances easier.

2. **Startup initialization (runs once):**
   ```python
   vector_store = DataIngestor().ingest(load_existing=True)
   rag_chain = RAGChainBuilder(vector_store).build_chain()
   ```
   - Connects to AstraDB (doesn't re-upload data, just connects)
   - Builds the full RAG chain and keeps it in memory
   - This means the first request is fast — everything is pre-loaded

3. **`GET /` — Home route:**
   - Increments the Prometheus request counter
   - Renders the chat UI HTML page

4. **`POST /get` — Chat API endpoint:**
   - Receives the user's message from the form field `msg`
   - Invokes the RAG chain with:
     - `{"input": user_input}` — the question
     - `session_id: "user-session"` — a fixed session ID (note: this means all users share the same conversation history in this implementation)
   - Returns just the answer text (not JSON — the frontend expects plain text)

5. **`GET /metrics` — Prometheus metrics endpoint:**
   - Returns Prometheus-formatted metrics data
   - Prometheus server periodically scrapes this endpoint to collect metrics
   - Currently tracks `http_requests_total` — how many times the home page has been visited

6. **`app.run(host="0.0.0.0", port=5000)`:**
   - `0.0.0.0` makes the server accessible from any network interface (important for Docker)
   - Port 5000 is Flask's default
   - `debug=True` enables hot-reloading during development

---

### Step 6: Frontend (Chat UI)

**File:** `templates/index.html`

The frontend is a single-page chat interface that communicates with the Flask backend using AJAX.

**Key components:**

1. **External dependencies:**
   - Bootstrap 4.1 — for responsive grid layout and card components
   - Font Awesome 5 — for the send button icon (location arrow)
   - jQuery 3.3 — for DOM manipulation and AJAX requests

2. **Chat layout structure:**
   ```
   ┌──────────────────────────────────────┐
   │  [Bot Avatar]  Flipkart Chatbot      │  ← Card Header
   │                Ask me anything!       │
   ├──────────────────────────────────────┤
   │                                      │
   │         (Messages appear here)       │  ← Card Body (scrollable)
   │                                      │
   ├──────────────────────────────────────┤
   │  [Type your message...    ] [Send ➤] │  ← Card Footer (input form)
   └──────────────────────────────────────┘
   ```

3. **JavaScript logic (jQuery):**
   ```javascript
   $("#messageArea").on("submit", function(event) {
       // 1. Get current time for timestamp
       // 2. Get user's message from text input
       // 3. Create user message HTML bubble (green, right-aligned)
       // 4. Append it to the chat body
       // 5. Clear the input field
       // 6. Send AJAX POST to /get with {msg: rawText}
       // 7. On response, create bot message HTML bubble (blue, left-aligned)
       // 8. Append bot message to chat body
       // 9. Prevent default form submission (no page reload)
   });
   ```

**File:** `static/style.css`

Dark-themed styling for the chat interface:

- **Background:** Dark linear gradient (`#1e1e2f` → `#222b3a`)
- **Card:** Dark gray (`#2c2f36`) with rounded corners and box shadow
- **Header/Footer:** Deep blue (`#003366`) — Flipkart brand-inspired
- **User messages:** Green bubbles (`#2e7d32`), right-aligned
- **Bot messages:** Blue bubbles (`#0056b3`), left-aligned
- **Send button:** Gold/yellow (`#ffb900`) with hover animation
- **Text input:** Dark gray background with white text, yellow cursor caret
- **Responsive:** On mobile screens (<576px), the card goes full-height and message bubbles expand wider

---

## ⚙ Configuration & Environment Variables

**File:** `flipkart/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RAG_MODEL = "llama-3.1-8b-instant"
```

**Environment variables you need to set** (create a `.env` file in the project root):

```env
ASTRA_DB_API_ENDPOINT=https://your-db-id-region.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:your-token-here
ASTRA_DB_KEYSPACE=default_keyspace
GROQ_API_KEY=gsk_your-groq-api-key
```

| Variable | Where to get it |
|----------|-----------------|
| `ASTRA_DB_API_ENDPOINT` | DataStax Astra Console → Your Database → Connect → API Endpoint |
| `ASTRA_DB_APPLICATION_TOKEN` | DataStax Astra Console → Your Database → Connect → Generate Token |
| `ASTRA_DB_KEYSPACE` | DataStax Astra Console → Your Database → Keyspaces tab |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys → Create |

**Hardcoded model configs:**
- `EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"` — Downloaded from HuggingFace Hub. One of the best open-source embedding models for English text.
- `RAG_MODEL = "llama-3.1-8b-instant"` — Meta's LLaMA 3.1 served through Groq's inference API.

---

## 🛠 Utility Modules (Logging & Exception Handling)

### Logger (`utils/logger.py`)

```python
import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
```

- Creates a `logs/` directory if it doesn't exist
- Creates daily log files named like `log_2026-03-29.log`
- Log format: `2026-03-29 22:15:30,123 - INFO - Your message here`
- `get_logger(name)` returns a named logger so you can see which module generated each log

### Custom Exception (`utils/custom_exception.py`)

```python
import sys

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
        return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"

    def __str__(self):
        return self.error_message
```

- Extends Python's built-in `Exception` class
- Automatically extracts the **file name** and **line number** where the error occurred using `sys.exc_info()`
- Produces detailed error messages like: `"Data loading failed | Error: FileNotFoundError | File: data_ingestion.py | Line: 22"`
- Makes debugging much easier compared to generic exceptions

---

## 🚀 How to Run This Project Locally

### Prerequisites
- Python 3.10+
- A DataStax Astra DB account (free tier available)
- A Groq API key (free tier available)

### Step-by-step:

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/flipkart_product_recommender.git
cd flipkart_product_recommender
```

**2. Create and activate a virtual environment:**
```bash
python -m venv flipvenv
# Windows:
flipvenv\Scripts\activate
# Linux/Mac:
source flipvenv/bin/activate
```

**3. Install the project as an editable package:**
```bash
pip install -e .
```
This runs `setup.py`, which reads `requirements.txt` and installs all dependencies including the `flipkart` package itself. The `-e` flag means "editable" — changes to source code take effect immediately without reinstalling.

**4. Create a `.env` file in the project root:**
```env
ASTRA_DB_API_ENDPOINT=https://your-endpoint.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:your-token
ASTRA_DB_KEYSPACE=default_keyspace
GROQ_API_KEY=gsk_your-groq-key
```

**5. Ingest data into AstraDB (only needed once):**
```bash
python -m flipkart.data_ingestion
```
This reads the CSV, embeds all 450 reviews, and uploads them to AstraDB. Takes a few minutes.

**6. Run the Flask app:**
```bash
python app.py
```

**7. Open your browser:**
Navigate to `http://localhost:5000` — you should see the chat interface. Try asking:
- *"Which is the best Bluetooth headset for bass?"*
- *"How is the battery life of BoAt Rockerz?"*
- *"Compare OnePlus Bullets and realme Buds"*

---

## 🐳 Docker Containerization

**File:** `DockerFile`

```dockerfile
# Start from Python 3.10 slim image (smaller than full image)
FROM python:3.10-slim

# Prevent Python from writing .pyc cache files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set /app as the working directory inside the container
WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . .

# Install Python dependencies via setup.py
RUN pip install --no-cache-dir -e .

# Document that the app uses port 5000
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]
```

**Build and run:**
```bash
# Build the Docker image
docker build -t flask-app:latest .

# Run the container (pass env variables)
docker run -p 5000:5000 \
  -e ASTRA_DB_API_ENDPOINT=your-endpoint \
  -e ASTRA_DB_APPLICATION_TOKEN=your-token \
  -e ASTRA_DB_KEYSPACE=your-keyspace \
  -e GROQ_API_KEY=your-key \
  flask-app:latest
```

**Why `--no-cache-dir`?** Prevents pip from caching downloaded packages inside the container, keeping the image smaller.

**Why `build-essential`?** Some Python packages (like `numpy`, which is a dependency of the embedding model) require C compilation during installation.

---

## ☸ Kubernetes Deployment

The project includes Kubernetes manifests for deploying the entire stack on a cluster (e.g., GKE, EKS, or Minikube).

### Flask App Deployment

**File:** `flask-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask
  template:
    spec:
      containers:
      - name: flask-container
        image: flask-app:latest
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 5000
        envFrom:
          - secretRef:
              name: llmops-secrets    # K8s Secret containing API keys
---
apiVersion: v1
kind: Service
metadata:
  name: flask-service
spec:
  type: LoadBalancer              # Exposes the service externally
  ports:
    - port: 80                     # External port
      targetPort: 5000             # Container port
```

**Key details:**
- **`envFrom: secretRef`**: Instead of hardcoding API keys, they're pulled from a Kubernetes Secret called `llmops-secrets`
- **`LoadBalancer`** service type: Cloud providers automatically provision an external IP/load balancer
- External port 80 maps to internal container port 5000

**Creating the Kubernetes secret:**
```bash
kubectl create secret generic llmops-secrets \
  --from-literal=ASTRA_DB_API_ENDPOINT=your-endpoint \
  --from-literal=ASTRA_DB_APPLICATION_TOKEN=your-token \
  --from-literal=ASTRA_DB_KEYSPACE=your-keyspace \
  --from-literal=GROQ_API_KEY=your-key
```

**Deploying:**
```bash
kubectl apply -f flask-deployment.yaml
```

---

## 📊 Monitoring with Prometheus & Grafana

### How the Monitoring Pipeline Works

```
Flask App (/metrics) ──scrape──▶ Prometheus ──data source──▶ Grafana Dashboard
```

1. **Flask app exposes metrics** at `GET /metrics` using the `prometheus_client` library
2. **Prometheus scrapes** this endpoint every 15 seconds
3. **Grafana reads** from Prometheus and displays beautiful dashboards

### Prometheus Setup

**Config:** `prometheus/prometheus-configmap.yaml`

```yaml
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s        # How often Prometheus pulls metrics

    scrape_configs:
      - job_name: 'prometheus'     # Prometheus scrapes itself too
        static_configs:
          - targets: ['localhost:9090']

      - job_name: 'flask-app'      # Scrape the Flask app
        metrics_path: /metrics     # The endpoint to scrape
        static_configs:
          - targets: ['34.42.228.136:5000']   # Flask app's IP:port
```

**Deployment:** `prometheus/prometheus-deployment.yaml`
- Runs Prometheus in a container using the official `prom/prometheus` image
- Mounts the config file from the ConfigMap
- Exposes on NodePort `32001` (access via `http://<node-ip>:32001`)

**Deploy Prometheus:**
```bash
kubectl create namespace monitoring
kubectl apply -f prometheus/prometheus-configmap.yaml
kubectl apply -f prometheus/prometheus-deployment.yaml
```

### Grafana Setup

**Deployment:** `grafana/grafana-deployment.yaml`
- Runs Grafana using the official `grafana/grafana` image
- Deployed in the `monitoring` namespace
- Exposes on NodePort `32000` (access via `http://<node-ip>:32000`)

**Deploy Grafana:**
```bash
kubectl apply -f grafana/grafana-deployment.yaml
```

**After deployment:**
1. Access Grafana at `http://<node-ip>:32000`
2. Default login: `admin` / `admin`
3. Add Prometheus as a data source: URL = `http://prometheus-service:9090`
4. Create dashboards to visualize `http_requests_total` and other metrics

### Current Metric Tracked

The app currently tracks one metric:

```python
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")
```

- **Type:** Counter (only goes up)
- **Name:** `http_requests_total`
- **Incremented:** Every time someone visits the home page (`/`)
- **Use cases:** Track total traffic, calculate request rate with PromQL: `rate(http_requests_total[5m])`

---

## 🔄 How the Conversation Flow Works (Start to Finish)

Let's trace a complete user interaction through the entire system:

### 1. User opens `http://localhost:5000`
- Flask's `index()` route is triggered
- Prometheus counter increments
- `index.html` is rendered and sent to the browser
- User sees the dark-themed chat interface

### 2. User types: *"Which headset has the best bass?"* and hits Send
- jQuery intercepts the form submission (prevents page reload)
- A green message bubble appears on the right side with the user's text
- jQuery sends an AJAX `POST` to `/get` with `msg=Which headset has the best bass?`

### 3. Flask receives the request
- Extracts `user_input` from the form data
- Invokes `rag_chain.invoke()` with the input and session ID

### 4. RAG Chain processes the query
- **History check:** Is there prior conversation? If yes, the context prompt rewrites the question incorporating history. If no, the question passes through as-is.
- **Embedding:** The question is converted to a 768-dim vector using BGE
- **Retrieval:** AstraDB performs cosine similarity search, returns the top 3 most relevant reviews (e.g., reviews mentioning "bass," "deep sound," "low-end")
- **Generation:** The QA prompt combines the 3 reviews + question + history → sends to Groq LLaMA 3.1 → LLM generates a concise answer like: *"Based on customer reviews, the BoAt Rockerz 235v2 and OnePlus Bullets Wireless Z Bass Edition are frequently praised for deep bass. The Rockerz 235v2 has 'very high bass' according to multiple reviewers, while the Bass Edition lives up to its name with 'powerful bass response.'"*
- **History update:** Both the question and answer are saved to session history

### 5. Response returns to the browser
- Flask returns the plain text answer
- jQuery creates a blue message bubble on the left side with the bot's response
- User can now ask follow-up questions that reference the conversation context

### 6. Follow-up question: *"How is its battery life?"*
- "Its" refers to a headset mentioned before — the history-aware retriever recognizes this
- The context prompt rewrites it to: *"How is the battery life of the BoAt Rockerz 235v2?"*
- This rewritten query searches the vector database more accurately
- The cycle repeats with context-aware responses

---

## 📦 Package Setup

**File:** `setup.py`

```python
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="FLIPKART RECOMMENDER",
    version="0.1",
    author="Owais",
    packages=find_packages(),
    install_requires=requirements,
)
```

- Makes the project installable via `pip install -e .`
- `find_packages()` automatically discovers the `flipkart/` and `utils/` packages (any directory with `__init__.py`)
- Reads `requirements.txt` to install all dependencies automatically
- The `-e` (editable) flag means you can modify source code without reinstalling

**Dependencies (`requirements.txt`):**

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.3.25 | Core orchestration framework for RAG chain |
| `langchain-astradb` | 0.6.0 | AstraDB vector store integration |
| `langchain-huggingface` | 0.1.0 | HuggingFace embedding model integration |
| `langchain-groq` | 0.2.4 | Groq LLM provider integration |
| `langchain-community` | 0.3.24 | Community integrations (ChatMessageHistory) |
| `datasets` | 2.20.0 | HuggingFace datasets library |
| `pypdf` | 4.3.1 | PDF processing (available but not used in current pipeline) |
| `python-dotenv` | 1.0.1 | Load `.env` files as environment variables |
| `pandas` | 2.2.2 | CSV reading and DataFrame operations |
| `flask` | 3.0.3 | Web server framework |
| `prometheus_client` | 0.20.0 | Expose app metrics in Prometheus format |
| `huggingface_hub` | 1.3.0 | Download models from HuggingFace Hub |
| `fsspec` | 2024.5.0 | Filesystem abstraction (dependency of datasets) |

---

> **You've now read the complete guide.** Every file, every function, every design decision has been explained. You should now be able to understand, modify, deploy, and extend this project with full confidence.