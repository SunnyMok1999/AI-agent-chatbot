import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { createVectorStore } from '../api/services/vectorstore.service.js';

dotenv.config();

const chunkSize = parseInt(process.env.CHUNK_SIZE || '1000');
const chunkOverlap = parseInt(process.env.CHUNK_OVERLAP || '200');

function resolveInputFiles(args: string[]): string[] {
  if (!args.length) {
    throw new Error('No files provided. Usage: npm run ingest:file -- /absolute/path/book.pdf [/absolute/path/notes.txt]');
  }

  return args.map((filePath) => path.resolve(filePath));
}

async function loadDocuments(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();

  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }

  if (ext === '.pdf') {
    const loader = new PDFLoader(filePath);
    return loader.load();
  }

  if (ext === '.txt' || ext === '.md') {
    const loader = new TextLoader(filePath);
    return loader.load();
  }

  throw new Error(`Unsupported file type: ${ext}. Allowed: .pdf, .txt, .md`);
}

async function ingestFile(filePath: string) {
  console.log(`📄 Loading: ${filePath}`);
  const docs = await loadDocuments(filePath);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });

  const chunks = await splitter.splitDocuments(docs);
  console.log(`🧩 Chunks: ${chunks.length}`);

  await createVectorStore(chunks);
  console.log(`✅ Ingested: ${path.basename(filePath)}`);
}

async function main() {
  const files = resolveInputFiles(process.argv.slice(2));

  for (const filePath of files) {
    await ingestFile(filePath);
  }

  console.log('🎉 All files ingested successfully.');
}

main().catch((error) => {
  console.error('❌ Ingestion failed:', error?.message || error);
  process.exit(1);
});
