import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';

dotenv.config();

const nvidiaApiKey = process.env.NVIDIA_API_KEY || '';
const nvidiaBaseUrl = process.env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1';
const nvidiaModel = process.env.NVIDIA_MODEL || 'meta/llama-3.1-8b-instruct';
const nvidiaVlmModel = process.env.NVIDIA_VLM_MODEL || '';

const openRouterApiKey = process.env.OPEN_ROUTER_API_KEY || '';
const openRouterModel = process.env.OPEN_ROUTER_MODEL || 'meta/llama-3.1-8b-instruct';
const openRouterBaseUrl = process.env.OPEN_ROUTER_BASE_URL || 'https://openrouter.ai/api/v1';

const customGetTokenIds = (text: string): number[] => {
  // Lightweight estimator to avoid tiktoken "Unknown model" warnings for non-OpenAI model IDs.
  const pieces = (text || '').split(/\s+/).filter(Boolean);
  return pieces.map((_x, i) => i);
};

const originalWarn = console.warn.bind(console);
console.warn = (...args: any[]) => {
  const msg = String(args?.[0] || '');
  if (msg.includes('Failed to calculate number of tokens, falling back to approximate count')) {
    return;
  }
  originalWarn(...args);
};

// Use NVIDIA NIM by default if configured, else fallback to OpenRouter
let llm: ChatOpenAI;

if (nvidiaApiKey) {
  console.log('✅ Using NVIDIA NIM LLM service');
  llm = new ChatOpenAI({
    openAIApiKey: nvidiaApiKey,
    configuration: {
      baseURL: nvidiaBaseUrl,
    },
    modelName: nvidiaModel,
    customGetTokenIds,
    streaming: true,
    temperature: 0.1,
  } as any);
} else if (openRouterApiKey) {
  console.log('✅ Using OpenRouter LLM service');
  llm = new ChatOpenAI({
    openAIApiKey: openRouterApiKey,
    configuration: {
      baseURL: openRouterBaseUrl,
    },
    modelName: openRouterModel,
    customGetTokenIds,
    streaming: true,
    temperature: 0.1,
  } as any);
} else {
  console.warn('⚠️ No LLM API key found (NVIDIA_API_KEY or OPEN_ROUTER_API_KEY). Chat will not work.');
  // Initialize with dummy to avoid crashes, though calls will fail
  llm = new ChatOpenAI({
    openAIApiKey: 'dummy',
    modelName: 'dummy',
    customGetTokenIds,
  } as any);
}

export const nvidiaLlm = llm; // Keep name for backward compatibility in RAG service

export const nvidiaVlm =
  nvidiaApiKey && nvidiaVlmModel
    ? new ChatOpenAI({
        openAIApiKey: nvidiaApiKey,
        configuration: {
          baseURL: nvidiaBaseUrl,
        },
        modelName: nvidiaVlmModel,
        customGetTokenIds,
        streaming: true,
        temperature: 0.1,
      } as any)
    : null;

export const hasNvidiaVlm = Boolean(nvidiaVlm);

export async function invokeNvidiaVlmWithImage(question: string, imageDataUrl: string): Promise<string> {
  if (!nvidiaApiKey || !nvidiaVlmModel) {
    throw new Error('NVIDIA VLM is not configured. Set NVIDIA_VLM_MODEL in .env.');
  }

  const endpoint = `${nvidiaBaseUrl.replace(/\/$/, '')}/chat/completions`;
  const payload = {
    model: nvidiaVlmModel,
    temperature: 0.1,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `You are an expert Mathematics assistant. Read the uploaded image and answer the question directly.\nQuestion: ${question}`,
          },
          {
            type: 'image_url',
            image_url: {
              url: imageDataUrl,
            },
          },
        ],
      },
    ],
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`NVIDIA VLM request failed (${response.status}): ${body}`);
  }

  const data: any = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content === 'string') return content;
  return JSON.stringify(content ?? '');
}
export default nvidiaLlm;
