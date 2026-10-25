# Autonomous Agentic AI Proxy 🧠⚡

An autonomous, stateful AI proxy that intercepts incoming messages, augments them with context via a ChromaDB knowledge graph (Agentic RAG), and seamlessly bridges communication between a target client (via WhatsApp) and a human operator (via Telegram). 

Built with system resilience in mind, it features a **Zero-Token Local Fallback Engine**, **Dynamic API Key Rotation**, and an **Autonomous Ghost Worker** to gracefully handle upstream server outages without dropping the communication thread.

## 🌟 Key Architecture & Features

*   **Agentic RAG (ChromaDB):** Ingests historical chat exports to build a dense semantic memory space. The pipeline uses `k=3` retrieval to inject past context and core facts into the LLM prompt, ensuring responses match exact tone, formatting, and historical continuity.
*   **Human-in-the-Loop (HITL) via Telegram:** Incoming messages trigger a Telegram card with an AI-generated draft. Operators can approve, edit, or regenerate the draft via inline buttons before it routes back to WhatsApp.
*   **Persistent Autopilot Toggle:** Operators can send `/toggle_mode` in Telegram to instantly bypass human approval. The webhook reads this persistent SQLite configuration and routes AI responses directly to the client in real-time.
*   **Zero-Token Local Fallback Engine:** Intercepts `503 UNAVAILABLE` and `429 QUOTA EXHAUSTED` errors *before* they crash the pipeline. It dispatches localized, context-aware stall messages (e.g., "Network is lagging" or "Going to sleep") to maintain the illusion of presence without consuming LLM tokens.
*   **Autonomous Ghost Worker:** When a 503 error occurs, the server spawns an asynchronous background worker that polls the API every 60 seconds. Once server capacity recovers, the worker autonomously generates the queued draft and pushes it to the routing layer.
*   **Stateful API Key Rotation:** Features an exponential backoff engine that automatically rotates through a pool of Gemini API keys to evade free-tier rate limits.

## 🛠️ Prerequisites

*   Python 3.10+
*   Meta Developer Account (WhatsApp Business API)
*   Telegram Bot Token (via BotFather)
*   Google Gemini API Key(s)

## 🚀 Setup & Installation

### 1. Clone the Repository
`git clone https://github.com/s1dddpatty/autonomous-agentic-ai-proxy.git`
`cd autonomous-agentic-ai-proxy`
`python -m venv venv`
`source venv/bin/activate`  *(On Windows: `venv\Scripts\activate`)*
`pip install -r requirements.txt`

### 2. Configure Environment Variables
Copy `.env.example` to a new file named `.env`:
```bash
cp .env.example .env
```

Update `TARGET_CONTACT_NAME` and `TARGET_PHONE_NUMBER` with the person you are setting this proxy up for:
```env
# Contact Configuration
TARGET_CONTACT_NAME="PersonName"
TARGET_PHONE_NUMBER="919876543210"
```
Populate your Gemini API key(s), Meta WhatsApp Business credentials, and Telegram Bot tokens.

*Note: If you have a paid API tier, set `ENABLE_KEY_ROTATION=False` in your `.env` to bypass the multi-key backoff loop.*

### 3. Data Ingestion (Vector Database Training)
Place your historical chat export (`_chat.txt`) in the `data/` folder and run the ingestion scripts to build the local ChromaDB vector space:
`python scripts/ingest_chats.py`
`python scripts/inject_core_facts.py`

### 4. Start the Webhook Daemon
Launch the FastAPI server and Telegram polling client:
`python run_webhook.py`

*Expose your local port (default 8000) using ngrok or deploy directly to a cloud provider like GCP/AWS.*

## 🔌 Modularity & Gateway Customization

This project utilizes strict **Separation of Concerns**. The `TARGET_PHONE_NUMBER` and Meta API configurations are treated solely as I/O endpoints. 

To adapt this proxy for a different platform (e.g., Instagram Direct, Slack, Discord), simply replace `src/gateways/whatsapp_api.py` with your desired gateway logic. The core orchestration, state graph, and RAG pipelines are completely content-agnostic.

---
*Built for scale, resilience, and highly personalized AI routing.*
