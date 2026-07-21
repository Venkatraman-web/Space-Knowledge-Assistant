# 🚀 Space Facts RAG Chatbot

An AI-powered chatbot that answers space-related questions using **Retrieval-Augmented Generation (RAG)**. Instead of relying solely on a language model, the chatbot retrieves the most relevant information from a curated knowledge base of space facts stored in a **ChromaDB vector database**, enabling accurate and context-aware responses.

---

## ✨ Features

* 🌌 Ask natural language questions about space and astronomy
* 🧠 Semantic search using vector embeddings
* 📚 Retrieval-Augmented Generation (RAG)
* 🤖 Supports multiple local LLMs via **Ollama**
* 🔍 Switch between different embedding models
* ⚡ Streaming responses for a smooth chat experience
* 📖 Displays retrieved reference chunks
* 📝 Shows the augmented prompt used for response generation
* 💻 Interactive Streamlit interface

---

## 🛠️ Tech Stack

| Category        | Technologies                         |
| --------------- | ------------------------------------ |
| Frontend        | Streamlit                            |
| Backend         | Python                               |
| LLM             | Ollama (Llama 2, Llama 3.2)          |
| Vector Database | ChromaDB                             |
| Embeddings      | Chroma Default, Nomic Embed Text     |
| Data Processing | Pandas                               |
| Dataset         | CSV-based Space Facts Knowledge Base |

---

## 📌 System Architecture

```text
                  Space Facts Dataset (CSV)
                           │
                           ▼
                  Generate Embeddings
                           │
                           ▼
                  Chroma Vector Database
                           │
                           ▼
                     User Question
                           │
                           ▼
                 Semantic Similarity Search
                           │
                           ▼
                  Retrieve Relevant Facts
                           │
                           ▼
                  Ollama (Llama 2 / 3.2)
                           │
                           ▼
                 Context-Aware Response
                           │
                           ▼
                    Streamlit Interface
```

---

## 📂 Project Structure

```text
Space-Facts-RAG/
│
├── chatbot.py          # RAG pipeline and LLM interaction
├── ui.py               # Streamlit interface
├── space_facts.csv     # Knowledge base
├── db/                 # Chroma persistent database
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Space-Facts-RAG.git
cd Space-Facts-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com

Pull the required models:

```bash
ollama pull llama3.2
ollama pull llama2
ollama pull nomic-embed-text
```

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run ui.py
```

---

## 📖 How It Works

1. A dataset containing space facts is loaded from a CSV file.
2. Each fact is converted into vector embeddings.
3. The embeddings are stored in a persistent ChromaDB collection.
4. When a user asks a question, the query is embedded.
5. ChromaDB retrieves the most semantically relevant facts.
6. The retrieved context is provided to the selected Ollama language model.
7. The model generates a concise, context-aware answer.
8. The application streams the response while displaying the retrieved references.

---

## 🎯 Example Workflow

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Search ChromaDB
      │
      ▼
Retrieve Relevant Facts
      │
      ▼
Augment Prompt
      │
      ▼
Llama 2 / Llama 3.2
      │
      ▼
Generate Response
      │
      ▼
Display Answer + References
```

---

## 💡 Future Improvements

* Support larger knowledge bases
* Multi-document retrieval
* Hybrid search (BM25 + Semantic Search)
* Source citations with similarity scores
* Conversational memory
* Voice-based interaction
* Query rewriting for improved retrieval
* Reranking retrieved documents
* FastAPI backend with React frontend
* Docker deployment

---

## 📚 Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* ChromaDB Vector Database
* Local LLM Inference
* Prompt Augmentation
* Streaming LLM Responses
* Knowledge Base Retrieval

---

## 🤝 Contributing

Contributions and suggestions are welcome. Feel free to fork the repository, open an issue, or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.
