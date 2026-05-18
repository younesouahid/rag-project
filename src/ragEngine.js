/**
 * ragEngine.js
 * Core RAG pipeline using LangChain.js + ChromaDB
 *
 * Pipeline:
 *  Upload → Load → Split → Embed (OpenAI) → ChromaDB
 *  Query  → Embed → Retrieve → Prompt → GPT → Answer + Citations
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

// ── LangChain imports ─────────────────────────────────────────────
import { RecursiveCharacterTextSplitter }   from '@langchain/textsplitters';
import { Chroma }                   from '@langchain/community/vectorstores/chroma';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { RetrievalQAChain }                  from 'langchain/chains';
import { PromptTemplate }                    from '@langchain/core/prompts';
import { Document }                          from '@langchain/core/documents';

// ── Document parsing ──────────────────────────────────────────────
import pdfParse   from 'pdf-parse/lib/pdf-parse.js';
import mammoth    from 'mammoth';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────
export const UPLOADS_DIR     = path.join(__dirname, '..', 'uploads');
export const VECTORSTORE_DIR = path.join(__dirname, '..', 'vectorstore');
export const INDEX_FILE      = path.join(VECTORSTORE_DIR, 'index.json');

const CHUNK_SIZE    = 800;
const CHUNK_OVERLAP = 150;
const RETRIEVE_K    = 5;

const SUPPORTED_EXTS = new Set(['.pdf', '.txt', '.md', '.docx']);

// Ensure dirs exist
[UPLOADS_DIR, VECTORSTORE_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ── Embeddings (OpenAI text-embedding-3-small) ───────────────────
function getEmbeddings() {
  return new OpenAIEmbeddings({
    model: 'text-embedding-3-small',
    apiKey: process.env.OPENAI_API_KEY,
  });
}

// ── LLM ───────────────────────────────────────────────────────────
function getLLM() {
  return new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0.2,
    maxTokens: 1500,
    apiKey: process.env.OPENAI_API_KEY,
  });
}

// ── RAG Prompt ────────────────────────────────────────────────────
const RAG_PROMPT = PromptTemplate.fromTemplate(`
You are a personalized research and learning assistant helping students understand course material.

Use ONLY the context below to answer the question.
- Provide a clear, structured answer.
- Use **bold** for key concepts.
- At the end, list every source used with format: [Source: filename, Page: N]
- If the context lacks the answer, say: "I don't have enough information in the uploaded documents to answer this."
- Suggest 1-2 related topics the student might explore.

Context:
{context}

Question: {question}

Answer:`);

// ── Index file (deduplication) ────────────────────────────────────
function loadIndex() {
  try { return JSON.parse(fs.readFileSync(INDEX_FILE, 'utf8')); }
  catch { return {}; }
}

function saveIndex(idx) {
  fs.writeFileSync(INDEX_FILE, JSON.stringify(idx, null, 2));
}

function fileHash(filePath) {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

// ── Document loaders ──────────────────────────────────────────────
async function loadFile(filePath, sourceName) {
  const ext = path.extname(filePath).toLowerCase();
  const docs = [];

  if (ext === '.pdf') {
    const buf = fs.readFileSync(filePath);
    const data = await pdfParse(buf);
    // Split by pages roughly
    const pages = data.text.split(/\f/).filter(p => p.trim());
    pages.forEach((pageText, i) => {
      docs.push(new Document({
        pageContent: pageText.trim(),
        metadata: { source: sourceName, page: i + 1 },
      }));
    });
    // If no form-feed separators, treat as single doc
    if (!docs.length) {
      docs.push(new Document({
        pageContent: data.text,
        metadata: { source: sourceName, page: 1 },
      }));
    }

  } else if (ext === '.docx') {
    const result = await mammoth.extractRawText({ path: filePath });
    docs.push(new Document({
      pageContent: result.value,
      metadata: { source: sourceName, page: 1 },
    }));

  } else if (ext === '.txt' || ext === '.md') {
    const text = fs.readFileSync(filePath, 'utf8');
    docs.push(new Document({
      pageContent: text,
      metadata: { source: sourceName, page: 1 },
    }));
  }

  return docs;
}

// ── Vector store singleton ─────────────────────────────────────────
let _vectorStore = null;

async function getVectorStore(create = false) {
  if (_vectorStore) return _vectorStore;

  const embeddings = getEmbeddings();
  const collectionName = 'course_docs';

  if (create) {
    _vectorStore = await Chroma.fromDocuments([], embeddings, {
      collectionName,
      url: process.env.CHROMA_URL || 'http://localhost:8000',
    });
  } else {
    _vectorStore = new Chroma(embeddings, {
      collectionName,
      url: process.env.CHROMA_URL || 'http://localhost:8000',
    });
  }
  return _vectorStore;
}

// ── Public API ─────────────────────────────────────────────────────

/**
 * Ingest a file into the vector store.
 */
export async function ingestFile(filePath, sourceName) {
  const ext = path.extname(filePath).toLowerCase();
  if (!SUPPORTED_EXTS.has(ext)) {
    throw new Error(`Unsupported file type "${ext}". Supported: ${[...SUPPORTED_EXTS].join(', ')}`);
  }

  const idx   = loadIndex();
  const hash  = fileHash(filePath);

  if (idx[hash]) {
    return { status: 'skipped', message: `"${sourceName}" is already indexed.`, chunks: 0 };
  }

  // Load
  const rawDocs = await loadFile(filePath, sourceName);
  if (!rawDocs.length) throw new Error('Could not extract text from file.');

  // Split
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:    CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators:   ['\n\n', '\n', '. ', ' ', ''],
  });
  const chunks = await splitter.splitDocuments(rawDocs);

  // Embed + store
  const embeddings = getEmbeddings();
  const vs = await Chroma.fromDocuments(chunks, embeddings, {
    collectionName: 'course_docs',
    url: process.env.CHROMA_URL || 'http://localhost:8000',
  });
  _vectorStore = vs;

  // Update index
  idx[hash] = { source: sourceName, filePath, chunks: chunks.length, pages: rawDocs.length };
  saveIndex(idx);

  return {
    status:  'indexed',
    message: `"${sourceName}" indexed successfully.`,
    chunks:  chunks.length,
    pages:   rawDocs.length,
  };
}

/**
 * Run a RAG query and return { answer, citations }.
 */
export async function ragQuery(question) {
  const idx = loadIndex();
  if (!Object.keys(idx).length) {
    throw new Error('No documents indexed. Upload course materials first.');
  }

  const embeddings   = getEmbeddings();
  const vectorStore  = new Chroma(embeddings, {
    collectionName: 'course_docs',
    url: process.env.CHROMA_URL || 'http://localhost:8000',
  });

  const retriever = vectorStore.asRetriever({ k: RETRIEVE_K });
  const llm       = getLLM();

  const chain = RetrievalQAChain.fromLLM(llm, retriever, {
    returnSourceDocuments: true,
    prompt: RAG_PROMPT,
  });

  const result = await chain.invoke({ query: question });

  // Parse citations from source docs
  const seen = new Set();
  const citations = [];
  for (const doc of (result.sourceDocuments || [])) {
    const { source = 'Unknown', page = 'N/A' } = doc.metadata;
    const key = `${source}|${page}`;
    if (!seen.has(key)) {
      seen.add(key);
      citations.push({
        source,
        page,
        snippet: doc.pageContent.slice(0, 220).trim() + '…',
      });
    }
  }

  return { answer: result.text, citations };
}

/**
 * List all indexed sources.
 */
export function listSources() {
  const idx = loadIndex();
  return Object.values(idx).map(({ source, chunks, pages }) => ({ source, chunks, pages }));
}

/**
 * Remove a source from the index tracking.
 */
export function deleteSource(sourceName) {
  const idx = loadIndex();
  const entry = Object.entries(idx).find(([, v]) => v.source === sourceName);
  if (!entry) return false;
  delete idx[entry[0]];
  saveIndex(idx);
  return true;
}

export function isIndexEmpty() {
  return Object.keys(loadIndex()).length === 0;
}

export { SUPPORTED_EXTS };
