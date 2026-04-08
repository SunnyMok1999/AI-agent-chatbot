import { Chroma } from '@langchain/community/vectorstores/chroma';
import { OpenAIEmbeddings } from '@langchain/openai';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import dotenv from 'dotenv';

dotenv.config();

const collectionName = process.env.CHROMA_COLLECTION_NAME || 'ml_documents';
const embeddingProvider = process.env.EMBEDDING_PROVIDER || 'openai';
const chromaUrl = process.env.CHROMA_URL || 'http://localhost:8000';

let embeddings: any;

if (embeddingProvider === 'huggingface') {
  console.log('✅ Using Hugging Face free embeddings service');
  embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: process.env.EMBEDDING_MODEL || 'BAAI/bge-base-en-v1.5',
  });
} else {
  console.log('✅ Using OpenAI embeddings service');
  embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
  });
}

let vectorStore: Chroma | null = null;

export const getVectorStore = async () => {
  if (vectorStore) return vectorStore;

  vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: collectionName,
    url: chromaUrl,
  }).catch(async (err) => {
    console.warn('Could not find existing Chroma collection, creating new one...');
    return null;
  });

  return vectorStore;
};

export const createVectorStore = async (docs: any[]) => {
  vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: collectionName,
    url: chromaUrl,
  });
  return vectorStore;
};

export default { getVectorStore, createVectorStore };
