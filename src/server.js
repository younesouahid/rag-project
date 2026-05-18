/**
 * server.js  —  Express API for the RAG Learning Assistant
 *
 * Routes:
 *   GET    /                       → Serve web UI
 *   GET    /api/health             → Status check
 *   GET    /api/sources            → List indexed sources
 *   DELETE /api/sources/:name      → Remove a source
 *   POST   /api/upload             → Upload & index a file
 *   POST   /api/query              → Ask a question
 */

import 'dotenv/config';
import express    from 'express';
import cors       from 'cors';
import multer     from 'multer';
import path       from 'path';
import fs         from 'fs';
import { fileURLToPath } from 'url';

import {
  ingestFile,
  ragQuery,
  listSources,
  deleteSource,
  isIndexEmpty,
  UPLOADS_DIR,
  SUPPORTED_EXTS,
} from './ragEngine.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.join(__dirname, '..');

// ── App setup ─────────────────────────────────────────────────────
const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(ROOT, 'public')));

// ── Multer (file upload) ──────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename:    (req, file, cb) => {
    const unique = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const safe   = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${unique}_${safe}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (SUPPORTED_EXTS.has(ext)) return cb(null, true);
    cb(new Error(`Unsupported type "${ext}". Allowed: ${[...SUPPORTED_EXTS].join(', ')}`));
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
});

// ── Routes ────────────────────────────────────────────────────────

// Health
app.get('/api/health', (req, res) => {
  const sources = listSources();
  res.json({
    status:         'ok',
    sources_indexed: sources.length,
    ready:          sources.length > 0,
  });
});

// List sources
app.get('/api/sources', (req, res) => {
  res.json({ sources: listSources() });
});

// Delete source
app.delete('/api/sources/:name', (req, res) => {
  const removed = deleteSource(decodeURIComponent(req.params.name));
  if (removed) return res.json({ status: 'removed' });
  res.status(404).json({ error: 'Source not found' });
});

// Upload & index
app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file provided' });

  // Original name (without the unique prefix we added)
  const originalName = req.file.originalname;
  const savedPath    = req.file.path;

  try {
    const result = await ingestFile(savedPath, originalName);
    res.json(result);
  } catch (err) {
    // Clean up the saved file on error
    fs.unlink(savedPath, () => {});
    res.status(500).json({ error: err.message });
  }
});

// Query
app.post('/api/query', async (req, res) => {
  const question = (req.body?.question || '').trim();
  if (!question) return res.status(400).json({ error: 'No question provided' });

  if (isIndexEmpty()) {
    return res.status(400).json({
      error: 'No documents indexed. Upload course materials first.',
    });
  }

  try {
    const result = await ragQuery(question);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Fallback → SPA index
app.get('*', (req, res) => {
  res.sendFile(path.join(ROOT, 'public', 'index.html'));
});

// ── Error handler ─────────────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('[Error]', err.message);
  res.status(err.status || 500).json({ error: err.message });
});

// ── Start ─────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🎓  ScholarAI RAG Server`);
  console.log(`    http://localhost:${PORT}`);
  console.log(`    Sources indexed: ${listSources().length}\n`);
});
