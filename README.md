# Agile MCP Server POC

Welcome to the **Agile MCP Server**! If you're a new trainee, you're in the right place. This README will explain what this project is, how it works under the hood, and how you can run and interact with it.

---

## 🎯 What is this project?
This repository is a **Proof of Concept (POC)** built during an internship at **Agile S.r.l.** 

Its goal is to act as a bridge between hospital data and Large Language Models (LLMs). We achieve this by using the **Model Context Protocol (MCP)**, which allows AI interfaces (like Claude Desktop or custom Agile AI gateways) to fetch context from an external server securely. 

### The Problem It Solves
When a clinician asks an LLM (like ChatGPT or Claude) "What is patient P001's medical history?", the LLM inherently doesn't know because it doesn't have access to the hospital's private database. Furthermore, if you just hand clinical data to an LLM, it might **hallucinate** facts when summarizing. In healthcare, an AI confidently inventing clinical data is extremely dangerous.

### The Solution
This server solves both problems:
1. **Data Access (via MCP):** It exposes specific Python functions as "tools" that an AI can use to look up patient data on demand.
2. **Clinical Safety (Uncertainty Layer):** Instead of just giving the LLM's first answer, the server introduces a novel "Uncertainty Layer" that tests how confident the LLM actually is about its summary.

---

## 🧠 How the Uncertainty Layer Works
When a tool that relies on an LLM (like `get_patient_summary`) is called, here is what happens behind the scenes:

1. **Stochastic Sampling:** The server asks the LLM the exact same question **N times** with a non-zero "temperature". This forces the LLM to generate slightly different variations if it's unsure.
2. **Measuring Disagreement:** 
   - **For Structured Responses (Categorical):** We calculate the *Shannon Entropy*. If the model outputs "P001" all three times, entropy is 0 (100% confident). If it outputs three completely different answers, entropy is high (0% confident).
   - **For Free-text Summaries (Narrative):** We use **Sentence Embeddings** (a local AI model `all-MiniLM-L6-v2`) to measure the *Semantic Similarity* of the sentences. It compares the meanings of the texts (not just word-for-word). If the 3 variations mean the same thing, confidence is HIGH. If the variations contradict each other, confidence is LOW.
3. **Safety Thresholds:** The tool responds with an `UncertainResult` containing the best answer *and* a `confidence_level` (HIGH, MEDIUM, or LOW). If the confidence is LOW, the AI interface knows to warn the clinician rather than presenting the data as a definitive fact.

---

## 🏗️ Repository Structure

Here is a quick tour of the code:

* `server.py` — **The Entrypoint.** This file sets up the `fastmcp` server and registers all our functions as tools.
* `data/mock_store.py` — **The Database.** A mock JSON database containing synthetic patient data.
* `features/patient_lookup.py` — **Deterministic Tools.** Basic tools that just fetch data (no LLM involved, like getting a patient's age or status).
* `features/patient_summary.py` — **LLM Tools.** The complex tools that query the LLM and format discharge drafts.
* `workflow/uncertainty.py` — **The Brains.** Where the Shannon Entropy and Sentence Embedding math lives.
* `tests/test_features.py` — **The Safeguards.** Pytest tests to ensure the math and lookups work correctly.

---

## 🚀 How to Run the Server

As a trainee, getting the server running on your machine is your first milestone.

### 1. Prerequisites
Ensure you have Python 3.11+ installed.

### 2. Setup the Environment
Open your terminal and run:
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# OR on Mac/Linux: source venv/bin/activate

# Install all dependencies (fastmcp, openai, etc.)
pip install -e .
```

### 3. Environment Variables
You will need an OpenAI API key for the LLM features. Create a `.env` file in the root directory and add:
```env
OPENAI_API_KEY="sk-your-api-key"
```

### 4. Running the Server
You have a few ways to run the server depending on your client:

**For local CLI testing (stdio transport):**
```bash
python server.py
```

**For HTTP testing (SSE transport):**
```bash
fastmcp run server.py --transport sse --port 8000
```

### 5. Testing the Tools
The easiest way to test if your tools are working is using the `mcp-cli`:
```bash
pip install mcp-cli

# List available tools
mcp-cli --server "python server.py" tools list

# Call a specific tool
mcp-cli --server "python server.py" tools call get_patient_age '{"patient_id": "P001"}'
```

You can also run the automated tests to verify the code integrity:
```bash
pytest tests/ -v
```

---

Welcome aboard! Play around with the mock data, run the tests, and see how the uncertainty estimation behaves when you tweak the prompts!
