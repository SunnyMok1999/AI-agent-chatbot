import { Chroma } from '@langchain/community/vectorstores/chroma';
import { OpenAIEmbeddings } from '@langchain/openai';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { Document } from '@langchain/core/documents';
import { Embeddings } from '@langchain/core/embeddings';
import { ChromaClient, IncludeEnum } from 'chromadb';
import dotenv from 'dotenv';

dotenv.config();

const collectionName = process.env.CHROMA_COLLECTION_NAME || 'ml_documents';
const embeddingProvider = process.env.EMBEDDING_PROVIDER || 'openai';
const chromaUrl = process.env.CHROMA_URL || 'http://localhost:8000';
const huggingFaceEndpointUrl = process.env.HUGGINGFACE_ENDPOINT_URL || 'https://router.huggingface.co';
const embeddingBatchSize = parseInt(process.env.EMBEDDING_BATCH_SIZE || '20');
const localEmbeddingModel = process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2';

let embeddings: any;

class LocalHashEmbeddings extends Embeddings {
  private readonly dimensions: number;

  constructor(dimensions = 256) {
    super({});
    this.dimensions = dimensions;
  }

  private toVector(text: string): number[] {
    const vector = new Array(this.dimensions).fill(0);
    const tokens = (text || '').toLowerCase().match(/[a-z0-9_]+/g) || [];

    for (const token of tokens) {
      let hash = 2166136261;
      for (let i = 0; i < token.length; i++) {
        hash ^= token.charCodeAt(i);
        hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
      }
      const idx = Math.abs(hash) % this.dimensions;
      vector[idx] += 1;
    }

    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (norm > 0) {
      return vector.map((v) => v / norm);
    }
    return vector;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.map((doc) => this.toVector(doc));
  }

  async embedQuery(document: string): Promise<number[]> {
    return this.toVector(document);
  }
}

class LocalTransformersEmbeddings extends Embeddings {
  private readonly modelName: string;
  private extractorPromise: Promise<any> | null = null;
  private readonly fallbackEmbeddings: LocalHashEmbeddings;
  private warnedFallback = false;

  constructor(modelName: string) {
    super({});
    this.modelName = modelName;
    this.fallbackEmbeddings = new LocalHashEmbeddings(parseInt(process.env.LOCAL_EMBEDDING_DIM || '256'));
  }

  private async getExtractor() {
    if (!this.extractorPromise) {
      this.extractorPromise = (async () => {
        const { pipeline, env } = await import('@xenova/transformers');
        env.allowLocalModels = true;
        env.useBrowserCache = false;
        return await pipeline('feature-extraction', this.modelName);
      })();
    }

    return this.extractorPromise;
  }

  private async embedText(text: string): Promise<number[]> {
    try {
      const extractor = await this.getExtractor();
      const output = await extractor(text || '', { pooling: 'mean', normalize: true });
      return Array.from(output.data as Float32Array);
    } catch (error) {
      if (!this.warnedFallback) {
        console.warn(
          `Local transformer model load failed (${this.modelName}). Falling back to local hash embeddings.`,
          error,
        );
        this.warnedFallback = true;
      }
      return await this.fallbackEmbeddings.embedQuery(text || '');
    }
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    const vectors: number[][] = [];
    for (const doc of documents) {
      vectors.push(await this.embedText(doc));
    }
    return vectors;
  }

  async embedQuery(document: string): Promise<number[]> {
    return await this.embedText(document);
  }
}

function sanitizeMetadataValue(value: unknown): string | number | boolean {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(', ');
  }

  return JSON.stringify(value);
}

function sanitizeMetadata(metadata: Record<string, unknown> | undefined): Record<string, string | number | boolean> {
  const safe: Record<string, string | number | boolean> = {};
  if (!metadata) return safe;

  for (const [key, value] of Object.entries(metadata)) {
    if (value === null || value === undefined) continue;
    safe[key] = sanitizeMetadataValue(value);
  }

  return safe;
}

if (embeddingProvider === 'huggingface') {
  console.log('✅ Using Hugging Face free embeddings service');
  embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: process.env.EMBEDDING_MODEL || 'BAAI/bge-base-en-v1.5',
    endpointUrl: huggingFaceEndpointUrl,
  });
} else if (embeddingProvider === 'openai') {
  console.log('✅ Using OpenAI embeddings service');
  embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
  });
} else if (embeddingProvider === 'local-transformers') {
  console.log(`✅ Using local transformers embeddings service (${localEmbeddingModel})`);
  embeddings = new LocalTransformersEmbeddings(localEmbeddingModel);
} else {
  console.log('✅ Using local hash embeddings service');
  embeddings = new LocalHashEmbeddings(parseInt(process.env.LOCAL_EMBEDDING_DIM || '256'));
}

let vectorStore: Chroma | null = null;
let chromaClient: ChromaClient | null = null;

function getChromaClient() {
  if (!chromaClient) {
    chromaClient = new ChromaClient({ path: chromaUrl });
  }
  return chromaClient;
}

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
  const safeDocs: Document[] = docs.map((doc) =>
    new Document({
      pageContent: typeof doc?.pageContent === 'string' ? doc.pageContent : String(doc?.pageContent ?? ''),
      metadata: sanitizeMetadata(doc?.metadata),
    }),
  );

  if (!vectorStore) {
    vectorStore = await getVectorStore();
  }

  const batches: Document[][] = [];
  for (let i = 0; i < safeDocs.length; i += embeddingBatchSize) {
    batches.push(safeDocs.slice(i, i + embeddingBatchSize));
  }

  if (vectorStore) {
    for (const batch of batches) {
      try {
        await vectorStore.addDocuments(batch);
      } catch (error) {
        console.warn('Batch ingestion failed, retrying documents one-by-one...', error);
        for (const doc of batch) {
          await vectorStore.addDocuments([doc]);
        }
      }
    }
  } else {
    vectorStore = await Chroma.fromDocuments(batches[0] || safeDocs, embeddings, {
      collectionName: collectionName,
      url: chromaUrl,
    });

    for (let i = 1; i < batches.length; i++) {
      try {
        await vectorStore.addDocuments(batches[i]);
      } catch (error) {
        console.warn('Batch ingestion failed, retrying documents one-by-one...', error);
        for (const doc of batches[i]) {
          await vectorStore.addDocuments([doc]);
        }
      }
    }
  }

  return vectorStore;
};

export const queryVectorStore = async (query: string, topK = 3): Promise<Document[]> => {
  try {
    const client = getChromaClient();
    const collection = await client.getCollection({
      name: collectionName,
      embeddingFunction: {
        generate: async (texts: string[]) => embeddings.embedDocuments(texts),
      },
    });
    const queryEmbedding = await embeddings.embedQuery(query);

    const result = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: topK,
      include: [IncludeEnum.Documents, IncludeEnum.Metadatas, IncludeEnum.Distances],
    });

    const docs = result.documents?.[0] || [];
    const metadatas = result.metadatas?.[0] || [];

    return docs
      .map((doc, index) => {
        if (!doc) return null;
        return new Document({
          pageContent: String(doc),
          metadata: sanitizeMetadata((metadatas[index] as Record<string, unknown>) || {}),
        });
      })
      .filter((doc): doc is Document => Boolean(doc));
  } catch (error) {
    console.warn('Direct Chroma query failed:', error);
    return [];
  }
};

export default { getVectorStore, createVectorStore, queryVectorStore };
