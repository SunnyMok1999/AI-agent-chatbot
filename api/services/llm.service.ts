import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';

dotenv.config();

const nvidiaApiKey = process.env.NVIDIA_API_KEY || '';
const nvidiaBaseUrl = process.env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1';
const nvidiaModel = process.env.NVIDIA_MODEL || 'meta/llama-3.1-8b-instruct';

const openRouterApiKey = process.env.OPEN_ROUTER_API_KEY || '';
const openRouterModel = process.env.OPEN_ROUTER_MODEL || 'meta/llama-3.1-8b-instruct';
const openRouterBaseUrl = process.env.OPEN_ROUTER_BASE_URL || 'https://openrouter.ai/api/v1';

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
    streaming: true,
    temperature: 0.1,
  });
} else if (openRouterApiKey) {
  console.log('✅ Using OpenRouter LLM service');
  llm = new ChatOpenAI({
    openAIApiKey: openRouterApiKey,
    configuration: {
      baseURL: openRouterBaseUrl,
    },
    modelName: openRouterModel,
    streaming: true,
    temperature: 0.1,
  });
} else {
  console.warn('⚠️ No LLM API key found (NVIDIA_API_KEY or OPEN_ROUTER_API_KEY). Chat will not work.');
  // Initialize with dummy to avoid crashes, though calls will fail
  llm = new ChatOpenAI({
    openAIApiKey: 'dummy',
    modelName: 'dummy',
  });
}

export const nvidiaLlm = llm; // Keep name for backward compatibility in RAG service
export default nvidiaLlm;
