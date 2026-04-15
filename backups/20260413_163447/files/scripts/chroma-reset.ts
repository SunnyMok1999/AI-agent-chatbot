import dotenv from 'dotenv';
import { ChromaClient } from 'chromadb';

dotenv.config();

const chromaUrl = process.env.CHROMA_URL || 'http://localhost:8001';
const collectionName = process.env.CHROMA_COLLECTION_NAME || 'ml_documents';

async function main() {
  const client = new ChromaClient({ path: chromaUrl });

  const collections = await client.listCollections();
  const exists = collections.some((c) => c.name === collectionName);

  if (!exists) {
    console.log(`ℹ️ Collection '${collectionName}' does not exist. Nothing to reset.`);
    return;
  }

  await client.deleteCollection({ name: collectionName });
  console.log(`✅ Deleted collection '${collectionName}'.`);
  console.log('🧹 Clean re-ingest setup ready. Re-run ingestion scripts now.');
}

main().catch((error) => {
  console.error('❌ Failed to reset collection:', error?.message || error);
  process.exit(1);
});
