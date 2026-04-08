import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { createVectorStore } from '../api/services/vectorstore.service.js';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const seedDir = './data/seed';

async function seed() {
  console.log('🌱 Starting ingestion of seed documents...');
  
  const files = fs.readdirSync(seedDir);
  
  for (const file of files) {
    if (file.endsWith('.md') || file.endsWith('.txt')) {
      console.log(`Processing: ${file}`);
      const filePath = path.join(seedDir, file);
      const loader = new TextLoader(filePath);
      const docs = await loader.load();
      
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      
      const chunks = await splitter.splitDocuments(docs);
      await createVectorStore(chunks);
      console.log(`Ingested ${chunks.length} chunks from ${file}`);
    }
  }
  
  console.log('✅ Seed ingestion complete!');
}

seed().catch(err => {
  console.error('❌ Ingestion failed:', err);
  process.exit(1);
});
