import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { Document } from '@langchain/core/documents';
import { createVectorStore } from '../api/services/vectorstore.service.js';

dotenv.config();

type ManifestRow = {
  manifest_id?: string;
  agent_profile?: string;
  source_book?: string;
  author?: string;
  subject?: string;
  difficulty?: string;
  chapter?: string;
  domain?: string;
  tags?: string[] | string;
  source_type?: string;
  source_url?: string;
  license?: string;
  local_path?: string;
  local_paths?: string[] | string;
  enabled?: boolean | string;
  notes?: string;
};

const chunkSize = parseInt(process.env.CHUNK_SIZE || '1000', 10);
const chunkOverlap = parseInt(process.env.CHUNK_OVERLAP || '200', 10);
const defaultManifest = path.resolve('data/seed/ingest_manifest.json');

function printUsage() {
  console.log('Usage: npm run ingest:manifest -- [manifestPath]');
  console.log('Example: npm run ingest:manifest -- data/seed/ingest_manifest.json');
  console.log('Example: npm run ingest:manifest -- data/seed/ingest_manifest.csv');
}

function parseBool(value: unknown, fallback = true): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const v = value.trim().toLowerCase();
    if (v === 'true' || v === '1' || v === 'yes') return true;
    if (v === 'false' || v === '0' || v === 'no') return false;
  }
  return fallback;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === ',' && !inQuotes) {
      out.push(current.trim());
      current = '';
      continue;
    }

    current += ch;
  }

  out.push(current.trim());
  return out;
}

function normalizeTags(tags: unknown): string {
  if (Array.isArray(tags)) {
    return tags.map((t) => String(t).trim()).filter(Boolean).join(',');
  }
  if (typeof tags === 'string') {
    return tags
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
      .join(',');
  }
  return '';
}

function resolveLocalPaths(row: ManifestRow): string[] {
  const fromArray = row.local_paths;
  if (Array.isArray(fromArray)) {
    return fromArray.map((p) => String(p).trim()).filter(Boolean);
  }

  if (typeof fromArray === 'string' && fromArray.trim()) {
    const raw = fromArray.trim();

    // CSV may provide JSON text in one column: "[\"/a.pdf\",\"/b.pdf\"]"
    if (raw.startsWith('[') && raw.endsWith(']')) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return parsed.map((p) => String(p).trim()).filter(Boolean);
        }
      } catch {
        // Fall through to delimiter parsing.
      }
    }

    // Support common CSV delimiters for multiple files.
    return raw
      .split(/[;,|]/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  const single = String(row.local_path || '').trim();
  return single ? [single] : [];
}

function parseCsvManifest(content: string): ManifestRow[] {
  const rawLines = content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith('#'));

  if (!rawLines.length) return [];

  const headers = splitCsvLine(rawLines[0]);
  const rows: ManifestRow[] = [];

  for (let i = 1; i < rawLines.length; i++) {
    const values = splitCsvLine(rawLines[i]);
    const row: Record<string, unknown> = {};

    headers.forEach((h, idx) => {
      row[h] = values[idx] ?? '';
    });

    rows.push(row as ManifestRow);
  }

  return rows;
}

function loadManifest(manifestPath: string): ManifestRow[] {
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest not found: ${manifestPath}`);
  }

  const ext = path.extname(manifestPath).toLowerCase();
  const content = fs.readFileSync(manifestPath, 'utf-8');

  if (ext === '.json') {
    const parsed = JSON.parse(content);
    if (!Array.isArray(parsed)) {
      throw new Error('JSON manifest must be an array of objects');
    }
    return parsed as ManifestRow[];
  }

  if (ext === '.csv') {
    return parseCsvManifest(content);
  }

  throw new Error(`Unsupported manifest extension: ${ext}. Use .json or .csv`);
}

async function loadDocuments(filePath: string): Promise<Document[]> {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.pdf') {
    const loader = new PDFLoader(filePath);
    return loader.load();
  }

  if (ext === '.txt' || ext === '.md' || ext === '.tex') {
    const loader = new TextLoader(filePath);
    return loader.load();
  }

  throw new Error(`Unsupported local file type: ${ext}. Allowed: .pdf, .txt, .md, .tex`);
}

function getChunkMetadata(row: ManifestRow, absolutePath: string, chunkIndex: number): Record<string, unknown> {
  return {
    source: path.basename(absolutePath),
    source_path: absolutePath,
    chunk_index: chunkIndex,
    manifest_id: row.manifest_id || '',
    agent_profile: row.agent_profile || 'manager',
    source_book: row.source_book || '',
    author: row.author || '',
    subject: row.subject || '',
    difficulty: row.difficulty || '',
    chapter: row.chapter || '',
    domain: row.domain || 'general',
    tags: normalizeTags(row.tags),
    source_type: row.source_type || 'local',
    source_url: row.source_url || '',
    license: row.license || '',
    notes: row.notes || '',
  };
}

async function ingestRow(row: ManifestRow): Promise<number> {
  const enabled = parseBool(row.enabled, true);
  if (!enabled) return 0;

  const pathCandidates = resolveLocalPaths(row);
  if (!pathCandidates.length) {
    console.log(`⏭️  Skipping ${row.manifest_id || '(no manifest_id)'}: local_path/local_paths is empty`);
    return 0;
  }

  let total = 0;
  for (const p of pathCandidates) {
    const absolutePath = path.resolve(p);
    if (!fs.existsSync(absolutePath)) {
      console.log(`⏭️  Skipping ${row.manifest_id || '(no manifest_id)'}: file not found (${absolutePath})`);
      continue;
    }

    console.log(`📄 Loading ${absolutePath}`);
    const docs = await loadDocuments(absolutePath);

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
    });
    const chunks = await splitter.splitDocuments(docs);

    const enrichedChunks = chunks.map(
      (chunk, idx) =>
        new Document({
          pageContent: chunk.pageContent,
          metadata: {
            ...(chunk.metadata || {}),
            ...getChunkMetadata(row, absolutePath, idx),
          },
        }),
    );

    await createVectorStore(enrichedChunks);
    total += enrichedChunks.length;
    console.log(`✅ Ingested ${enrichedChunks.length} chunks [${row.manifest_id || path.basename(absolutePath)}]`);
  }

  return total;
}

async function main() {
  const arg = process.argv[2];
  if (arg === '--help' || arg === '-h') {
    printUsage();
    return;
  }

  const manifestPath = path.resolve(arg || defaultManifest);
  const rows = loadManifest(manifestPath);

  if (!rows.length) {
    console.log(`No entries found in manifest: ${manifestPath}`);
    return;
  }

  console.log(`🧾 Loaded ${rows.length} manifest entries from ${manifestPath}`);

  let totalChunks = 0;
  let processed = 0;

  for (const row of rows) {
    const count = await ingestRow(row);
    if (count > 0) processed += 1;
    totalChunks += count;
  }

  console.log(`🎉 Done. Processed entries: ${processed}, total chunks ingested: ${totalChunks}`);
}

main().catch((error) => {
  console.error('❌ Manifest ingestion failed:', error?.message || error);
  process.exit(1);
});
