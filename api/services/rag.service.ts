import { nvidiaLlm } from './llm.service.js';
import { getVectorStore } from './vectorstore.service.js';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { formatDocumentsAsString } from 'langchain/util/document';
import dotenv from 'dotenv';

dotenv.config();

const RAG_PROMPT = `
You are an expert Machine Learning and Deep Learning assistant.
Answer the question based ONLY on the following context.
If you don't have enough grounded information from the context, say "I don't have enough grounded information to answer this question accurately."
Do not invent facts. Prefer concise, source-backed answers.
Always cite your sources by referencing the document names and sections where applicable.

Context: {context}
Question: {question}

Answer:`;

export const processRagQuestion = async (question: string, onToken?: (token: string) => void) => {
  const vectorStore = await getVectorStore();
  if (!vectorStore) {
    throw new Error('Vector store not available. Please upload some documents first.');
  }

  const retriever = vectorStore.asRetriever({
    k: parseInt(process.env.TOP_K_RETRIEVAL || '5'),
    searchType: 'similarity',
  });

  const prompt = PromptTemplate.fromTemplate(RAG_PROMPT);

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: (input: { question: string }) => input.question,
    },
    prompt,
    nvidiaLlm,
    new StringOutputParser(),
  ]);

  if (onToken) {
    const stream = await chain.stream({ question });
    for await (const chunk of stream) {
      onToken(chunk);
    }
    return ''; // Stream handles the output
  } else {
    return await chain.invoke({ question });
  }
};

export default { processRagQuestion };
