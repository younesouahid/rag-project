# 🎓 ScholarAI — RAG Learning Assistant (Node.js / Express)

A **Retrieval-Augmented Generation (RAG)** application built with **LangChain.js**, **OpenAI**, and **Express** that acts as a personalized research and learning assistant — similar to Google NotebookLM. Upload your course materials and get answers with precise citations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│               Browser (public/index.html)                        │
│   Upload UI  ·  Chat Interface  ·  Source Manager               │
└────────────────────────┬─────────────────────────────────────────┘
                         │ HTTP (fetch API)
┌────────────────────────▼─────────────────────────────────────────┐
│              Express Server  (src/server.js)                     │
│                                                                  │
│  POST /api/upload    GET /api/sources    POST /api/query         │
│  DELETE /api/sources/:name               GET /api/health         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│               RAG Engine  (src/ragEngine.js)                     │
│                                                                  │
│  ┌──────────────┐  ┌─────────────────────────┐                  │
│  │  Doc Loaders │  │  LangChain.js Pipeline  │                  │
│  │  pdf-parse   │  │                         │                  │
│  │  mammoth     │→ │  RecursiveCharacter      │                  │
│  │  fs.read     │  │  TextSplitter            │                  │
│  └──────────────┘  │  (chunks: 800/150)       │                  │
│                    └────────────┬────────────┘                  │
│                                 │                                │
│              ┌──────────────────▼──────────────────┐            │
│              │       OpenAIEmbeddings               │            │
│              │   (text-embedding-3-small)           │            │
│              └──────────────────┬──────────────────┘            │
│                                 │                                │
│              ┌──────────────────▼──────────────────┐            │
│              │        ChromaDB (LangChain)          │            │
│              │     Persistent vector store          │            │
│              └──────────────────┬──────────────────┘            │
│                                 │                                │
│              ┌──────────────────▼──────────────────┐            │
│              │   RetrievalQAChain + PromptTemplate  │            │
│              │   ChatOpenAI (gpt-4o-mini)           │            │
│              └─────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

---

## LangChain.js Components Used

| Component                        | Package                     | Purpose                                          |
| -------------------------------- | --------------------------- | ------------------------------------------------ |
| `RecursiveCharacterTextSplitter` | `@langchain/textsplitters`  | Chunk docs at natural boundaries                 |
| `OpenAIEmbeddings`               | `@langchain/openai`         | Generate vector embeddings                       |
| `Chroma`                         | `@langchain/community`      | Persist & search vector store                    |
| `ChatOpenAI`                     | `@langchain/openai`         | LLM for answer generation                        |
| `RetrievalQAChain`               | `langchain/chains`          | End-to-end RAG pipeline                          |
| `PromptTemplate`                 | `@langchain/core/prompts`   | Structured RAG prompt with citation instructions |
| `Document`                       | `@langchain/core/documents` | Standard document wrapper                        |

---

## RAG Pipeline

### Ingestion

```
File → pdf-parse/mammoth → LangChain Documents → RecursiveCharacterTextSplitter
  → OpenAIEmbeddings (text-embedding-3-small) → ChromaDB (persisted)
```

### Query

```
Question → OpenAIEmbeddings → ChromaDB similarity search (top-5 chunks)
         → PromptTemplate (context + question) → ChatOpenAI (gpt-4o-mini)
         → Answer + structured citations (source, page, snippet)
```

---

## Quick Start

### 1. Install

```bash
npm install
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
npm start
# → http://localhost:3000
```

### Development mode (auto-restart)

```bash
npm run dev
```

---

## API Reference

| Method   | Endpoint             | Description                                        |
| -------- | -------------------- | -------------------------------------------------- |
| `GET`    | `/api/health`        | Server status & source count                       |
| `GET`    | `/api/sources`       | List all indexed documents                         |
| `DELETE` | `/api/sources/:name` | Remove a source                                    |
| `POST`   | `/api/upload`        | Upload file (`multipart/form-data`, field: `file`) |
| `POST`   | `/api/query`         | Ask a question `{ "question": "..." }`             |

### Example

```bash
# Upload a PDF
curl -X POST http://localhost:3000/api/upload \
  -F "file=@lecture1.pdf"

# Ask a question
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is backpropagation?"}'
```

---

## Project Structure

```
rag_node/
├── src/
│   ├── server.js        # Express app & all routes
│   └── ragEngine.js     # LangChain RAG pipeline
├── public/
│   └── index.html       # Full-featured web UI (single file)
├── uploads/             # Uploaded files (auto-created)
├── vectorstore/         # ChromaDB data + index.json (auto-created)
├── .env.example         # Environment template
├── .gitignore
├── package.json
└── README.md
```

---

## Supported File Types

| Extension | Parser      |
| --------- | ----------- |
| `.pdf`    | `pdf-parse` |
| `.docx`   | `mammoth`   |
| `.txt`    | Node `fs`   |
| `.md`     | Node `fs`   |

---

## Key Design Decisions

**Why `RecursiveCharacterTextSplitter`?**
Tries `\n\n → \n → . → space` in sequence, preserving paragraph and sentence structure before falling back to character-level splits.

**Why `text-embedding-3-small` embeddings?**
OpenAI's model is fast, cost-effective, and performs well for retrieval tasks.

**Why SHA-256 deduplication?**
Files are hashed before indexing so re-uploading the same document doesn't create duplicate chunks in ChromaDB.

**Why `RetrievalQAChain`?**
It handles the full retrieve → format → generate cycle with `returnSourceDocuments: true`, giving us the chunks for citation extraction.
