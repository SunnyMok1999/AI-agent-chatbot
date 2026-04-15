import { hasNvidiaVlm, invokeNvidiaVlmWithImage, nvidiaLlm } from './llm.service.js';
import { getVectorStore, queryVectorStore } from './vectorstore.service.js';
import { getLatestUploadContext, getLatestUploadImageContext } from './upload-context.service.js';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import dotenv from 'dotenv';

dotenv.config();

const RAG_PROMPT = `
You are a patient Mathematics tutor for university students.
Teach first, do not just give the final result.
Answer the question based ONLY on the following context.
If there is not enough grounded information in context, say: "I don't have enough grounded information to answer this question accurately."
Rules:
- Start with a short hint.
- Then show numbered step-by-step reasoning.
- If asked for direct answer, provide it clearly.
- End with: "Final answer: ..."
- Do not invent facts.
- Cite sources from the provided context.

Context: {context}
Question: {question}

Answer:`;

const GENERAL_PROMPT = `
You are a patient Mathematics tutor for university students.
Teach first, do not just give the final result.
Rules:
- Start with a short hint.
- Then show numbered step-by-step reasoning.
- If asked for direct answer, provide it clearly.
- End with: "Final answer: ..."
- If the question is outside mathematics, say so briefly and give helpful guidance.

Question: {question}

Answer:`;

const RECENT_UPLOAD_PROMPT = `
You are a patient Mathematics tutor for university students.
Use the recent uploaded file context to answer the question.
If the context contains a math problem, tutor the student:
- Start with a short hint.
- Then show numbered step-by-step reasoning.
- End with: "Final answer: ..."
If the answer is not present and cannot be derived from the context, say so briefly.

Context: {context}
Question: {question}

Answer:`;

const TUTOR_QUERY_PREFIX = 'Teach me like a tutor. Give hints first, then full solution if I ask.';

const REQUEST_TIMEOUT_MS = parseInt(process.env.RAG_TIMEOUT_MS || '45000');
const RETRIEVAL_TIMEOUT_MS = parseInt(process.env.RAG_RETRIEVAL_TIMEOUT_MS || '12000');
const LLM_TIMEOUT_MS = parseInt(process.env.RAG_LLM_TIMEOUT_MS || '25000');
const MAX_QUESTION_CHARS = parseInt(process.env.RAG_MAX_QUESTION_CHARS || '1200');
const ENABLE_DIRECT_LLM_FALLBACK = process.env.ENABLE_DIRECT_LLM_FALLBACK === 'true';
const ENABLE_SMALLTALK_FALLBACK = process.env.ENABLE_SMALLTALK_FALLBACK !== 'false';
const ENABLE_TUTOR_MODE = process.env.ENABLE_TUTOR_MODE !== 'false';
const ENABLE_HYBRID_RETRIEVAL = process.env.ENABLE_HYBRID_RETRIEVAL !== 'false';
const ENABLE_MATH_RERANKER = process.env.ENABLE_MATH_RERANKER !== 'false';
const RETRIEVAL_CANDIDATE_K = parseInt(process.env.RETRIEVAL_CANDIDATE_K || '16');

type MathIntent = {
  topic: 'algebra' | 'calculus' | 'linear_algebra' | 'vector_calculus' | 'proof' | 'general';
  mode: 'solve' | 'explain' | 'prove' | 'compute' | 'general';
  tags: string[];
};

function isSmallTalkQuestion(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hello|hey|yo|thanks|thank you|bye|good morning|good afternoon|good evening|how are you|who are you|what can you do)[!.?\s]*$/.test(
    normalized,
  );
}

function refersToUploadedDocument(text: string): boolean {
  return /\b(upload|uploaded|image|png|file|document|screenshot|photo)\b/i.test(text);
}

function refersToImage(text: string): boolean {
  return /\b(image|png|photo|screenshot|picture|diagram|graph)\b/i.test(text);
}

function asksForAnswer(text: string): boolean {
  return /\b(answer|solve|solution|result)\b/i.test(text);
}

function extractExplicitAnswer(content: string): string | null {
  const match = content.match(/\banswer\s*[:=-]\s*(.+?)(?:\n|$)/i);
  if (!match) return null;
  const answer = match[1].trim();
  return answer ? answer : null;
}

function evaluateSimpleMathExpression(text: string): string | null {
  const trimmed = text.trim().toLowerCase();

  // Accept prompts like "1-1", "1-1=?", "what is 2+2?"
  let extracted = trimmed
    .replace(/^what is\s+/, '')
    .replace(/^calculate\s+/, '')
    .replace(/\?+$/g, '')
    .replace(/=+$/g, '')
    .trim()
    .replace(/\s+/g, '');

  // Tolerate accidental trailing operators (e.g. "2-1-?").
  extracted = extracted.replace(/[+\-*/^]+$/g, '');

  if (!extracted) return null;
  if (!/^[0-9+\-*/().^a-z]+$/.test(extracted)) return null;
  if (!/[0-9]/.test(extracted) && !/\b(pi|e)\b/.test(extracted)) return null;
  if (extracted.length > 80) return null;

  // Accept shorthand function forms like "sin1" => "sin(1)"
  extracted = extracted.replace(
    /^(sin|cos|tan|asin|acos|atan|sqrt|abs|ln|log)(-?\d+(?:\.\d+)?)$/,
    '$1($2)',
  );

  try {
    if (!/^[0-9+\-*/().^a-z]+$/.test(extracted)) return null;

    const functionNames = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sqrt', 'abs'];

    let jsExpression = extracted.replace(/\^/g, '**');
    jsExpression = jsExpression.replace(/\bpi\b/g, 'Math.PI').replace(/\be\b/g, 'Math.E');
    jsExpression = jsExpression.replace(/\bln\(/g, 'Math.log(').replace(/\blog\(/g, 'Math.log10(');

    for (const fn of functionNames) {
      jsExpression = jsExpression.replace(new RegExp(`\\b${fn}\\(`, 'g'), `Math.${fn}(`);
    }

    const result = Function(`"use strict"; return (${jsExpression});`)();
    if (typeof result === 'number' && Number.isFinite(result)) {
      const rounded = Math.abs(result) < 1e-12 ? 0 : Number(result.toFixed(12));
      return `${extracted} = ${rounded}`;
    }
    return null;
  } catch {
    return null;
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, errorMessage: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(errorMessage)), timeoutMs);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

function classifyMathIntent(question: string): MathIntent {
  const q = question.toLowerCase();

  const isProof = /\b(prove|proof|show that|justify|why)\b/.test(q);
  const isSolve = /\b(solve|roots?|find x|equation|system of equations?)\b/.test(q);
  const isCompute = /\b(compute|evaluate|calculate|differentiate|integrate|derivative|integral)\b/.test(q);
  const isExplain = /\b(explain|intuition|concept|meaning|what is|why)\b/.test(q);

  let topic: MathIntent['topic'] = 'general';
  if (/\b(matrix|determinant|eigen|rank|nullspace|vector space|linear transformation)\b/.test(q)) {
    topic = 'linear_algebra';
  } else if (/\b(grad|gradient|divergence|curl|\bnabla\b|line integral|surface integral)\b/.test(q)) {
    topic = 'vector_calculus';
  } else if (/\b(derivative|integral|limit|continuity|chain rule|product rule|partial derivative|taylor)\b/.test(q)) {
    topic = 'calculus';
  } else if (/\b(polynomial|quadratic|factor|equation|inequality|algebra)\b/.test(q)) {
    topic = 'algebra';
  } else if (isProof) {
    topic = 'proof';
  }

  const mode: MathIntent['mode'] = isProof
    ? 'prove'
    : isSolve
      ? 'solve'
      : isCompute
        ? 'compute'
        : isExplain
          ? 'explain'
          : 'general';

  const tags = [topic, mode].filter((x) => x !== 'general');
  return { topic, mode, tags };
}

function expandQueryWithIntent(question: string, intent: MathIntent): string {
  if (!intent.tags.length) return question;

  const expansions: Record<string, string> = {
    algebra: 'equations roots factorization polynomial simplification',
    calculus: 'derivative integral limit chain rule product rule',
    linear_algebra: 'matrix determinant eigenvalue rank nullspace',
    vector_calculus: 'gradient divergence curl nabla line integral surface integral',
    proof: 'theorem lemma proof reasoning',
    solve: 'solve steps final answer',
    compute: 'calculation steps',
    explain: 'intuition concept explanation',
    prove: 'proof assumptions conclusion',
  };

  const extra = intent.tags.map((t) => expansions[t] || '').filter(Boolean).join(' ');
  return extra ? `${question} ${extra}` : question;
}

function tokenizeQuestion(question: string): string[] {
  return Array.from(
    new Set(
      (question.toLowerCase().match(/[a-z0-9_]{2,}/g) || []).filter(
        (t) => !['what', 'is', 'the', 'and', 'for', 'with', 'from', 'that', 'this'].includes(t),
      ),
    ),
  );
}

function scoreKeywordOverlap(content: string, queryTokens: string[]): number {
  if (!content || !queryTokens.length) return 0;
  const text = content.toLowerCase();
  let matched = 0;
  for (const token of queryTokens) {
    if (text.includes(token)) matched += 1;
  }
  return matched / queryTokens.length;
}

function scoreIntentMatch(
  metadata: Record<string, any> | undefined,
  content: string,
  intent: MathIntent,
): number {
  const tagsText = String(metadata?.tags || '').toLowerCase();
  const domain = String(metadata?.domain || '').toLowerCase();
  const body = content.toLowerCase();
  const signals = [tagsText, domain, body].join(' ');
  let score = 0;
  for (const tag of intent.tags) {
    if (signals.includes(tag.replace('_', ' ')) || signals.includes(tag)) score += 0.2;
  }
  return Math.min(1, score);
}

async function hybridRetrieve(question: string, topK: number): Promise<Array<{ pageContent?: string; metadata?: Record<string, any> }>> {
  const intent = classifyMathIntent(question);
  const expandedQuery = expandQueryWithIntent(question, intent);
  const candidateK = Math.max(topK, RETRIEVAL_CANDIDATE_K);

  const [docsA, docsB] = await Promise.all([
    queryVectorStore(question, candidateK),
    ENABLE_HYBRID_RETRIEVAL ? queryVectorStore(expandedQuery, candidateK) : Promise.resolve([]),
  ]);

  const dedup = new Map<string, { pageContent?: string; metadata?: Record<string, any> }>();
  [...docsA, ...docsB].forEach((doc) => {
    const key = `${doc.pageContent || ''}::${doc.metadata?.source || ''}`.slice(0, 500);
    if (!dedup.has(key)) dedup.set(key, doc);
  });

  let merged = Array.from(dedup.values());

  if (ENABLE_MATH_RERANKER) {
    const tokens = tokenizeQuestion(question);
    merged = merged
      .map((doc, index) => {
        const keywordScore = scoreKeywordOverlap(doc.pageContent || '', tokens);
        const intentScore = scoreIntentMatch(doc.metadata, doc.pageContent || '', intent);
        const recencyBoost = 1 / (index + 1);
        const totalScore = keywordScore * 0.6 + intentScore * 0.3 + recencyBoost * 0.1;
        return { doc, totalScore };
      })
      .sort((a, b) => b.totalScore - a.totalScore)
      .map((item) => item.doc);
  }

  return merged.slice(0, topK);
}

function applyTutorPrefix(question: string): string {
  const trimmed = question.trim();
  if (!trimmed) return trimmed;
  if (!ENABLE_TUTOR_MODE) return trimmed;
  if (trimmed.toLowerCase().startsWith(TUTOR_QUERY_PREFIX.toLowerCase())) {
    return trimmed;
  }
  return `${TUTOR_QUERY_PREFIX}\n\n${trimmed}`;
}

export const processRagQuestion = async (question: string, onToken?: (token: string) => void) => {
  if (typeof question !== 'string') {
    throw new Error('Question must be a string');
  }

  const normalizedQuestion = question.trim().slice(0, MAX_QUESTION_CHARS);
  const guidedQuestion = applyTutorPrefix(normalizedQuestion);
  if (!normalizedQuestion) {
    throw new Error('Question cannot be empty');
  }

  const runLlm = async (questionText: string, contextText?: string) => {
    if (contextText) {
      const prompt = PromptTemplate.fromTemplate(RAG_PROMPT);
      const chain = RunnableSequence.from([prompt, nvidiaLlm, new StringOutputParser()]);
      return await withTimeout(
        chain.invoke({ context: contextText, question: questionText }),
        Math.min(REQUEST_TIMEOUT_MS, LLM_TIMEOUT_MS),
        'LLM response timed out. Please try again.',
      );
    }

    const prompt = PromptTemplate.fromTemplate(GENERAL_PROMPT);
    const chain = RunnableSequence.from([prompt, nvidiaLlm, new StringOutputParser()]);
    return await withTimeout(
      chain.invoke({ question: questionText }),
      Math.min(REQUEST_TIMEOUT_MS, LLM_TIMEOUT_MS),
      'LLM response timed out. Please try again.',
    );
  };

  const runLlmWithRecentUploadContext = async (questionText: string, contextText: string) => {
    const prompt = PromptTemplate.fromTemplate(RECENT_UPLOAD_PROMPT);
    const chain = RunnableSequence.from([prompt, nvidiaLlm, new StringOutputParser()]);
    return await withTimeout(
      chain.invoke({ context: contextText, question: questionText }),
      Math.min(REQUEST_TIMEOUT_MS, LLM_TIMEOUT_MS),
      'LLM response timed out. Please try again.',
    );
  };

  const runVlmWithImage = async (questionText: string, imageDataUrl: string) => {
    const response = await withTimeout(
      invokeNvidiaVlmWithImage(questionText, imageDataUrl),
      Math.min(REQUEST_TIMEOUT_MS, LLM_TIMEOUT_MS),
      'Vision model response timed out. Please try again.',
    );
    return response;
  };

  const buildSourceLabel = (metadata: Record<string, any> | undefined, index: number) => {
    const source = metadata?.source || `document_${index + 1}`;
    const page = metadata?.loc?.pageNumber || metadata?.pageNumber || metadata?.locFrom;
    return page ? `${source} (page ${page})` : `${source}`;
  };

  const buildContext = (docs: Array<{ pageContent?: string; metadata?: Record<string, any> }>) => {
    return docs
      .map((doc, i) => {
        const label = buildSourceLabel(doc.metadata, i);
        return `[Source ${i + 1}] ${label}\n${doc.pageContent || ''}`;
      })
      .join('\n\n');
  };

  const buildSourcesFooter = (docs: Array<{ metadata?: Record<string, any> }>) => {
    const labels = docs.map((doc, i) => buildSourceLabel(doc.metadata, i));
    const unique = Array.from(new Set(labels));
    return unique.length ? `\n\nSources:\n- ${unique.join('\n- ')}` : '';
  };

  let answer = '';

  try {
    const simpleMathAnswer = evaluateSimpleMathExpression(normalizedQuestion);
    if (simpleMathAnswer) {
      if (onToken) {
        onToken(simpleMathAnswer);
        return '';
      }
      return simpleMathAnswer;
    }

    if (ENABLE_SMALLTALK_FALLBACK && isSmallTalkQuestion(normalizedQuestion)) {
      answer = await runLlm(guidedQuestion);
      if (onToken) {
        onToken(answer);
        return '';
      }
      return answer;
    }

    const latestImageUpload = getLatestUploadImageContext();
    if (latestImageUpload && refersToImage(normalizedQuestion) && !hasNvidiaVlm) {
      answer =
        'A vision model is not configured yet. Set NVIDIA_VLM_MODEL in .env to enable direct PNG/image question answering.';
      if (onToken) {
        onToken(answer);
        return '';
      }
      return answer;
    }

    if (latestImageUpload && refersToImage(normalizedQuestion) && hasNvidiaVlm) {
      answer = await runVlmWithImage(guidedQuestion, latestImageUpload.dataUrl);
      answer += `\n\nSources:\n- ${latestImageUpload.source}`;
      if (onToken) {
        onToken(answer);
        return '';
      }
      return answer;
    }

    const latestUpload = getLatestUploadContext();
    if (latestUpload && refersToUploadedDocument(normalizedQuestion)) {
      if (asksForAnswer(normalizedQuestion)) {
        const explicitAnswer = extractExplicitAnswer(latestUpload.content);
        if (explicitAnswer) {
          const uploadAnswer = `${explicitAnswer}\n\nSources:\n- ${latestUpload.source}`;
          if (onToken) {
            onToken(uploadAnswer);
            return '';
          }
          return uploadAnswer;
        }
      }

      const uploadContext = `[Recent Upload] ${latestUpload.source}\n${latestUpload.content}`;
      answer = await runLlmWithRecentUploadContext(guidedQuestion, uploadContext);
      answer += `\n\nSources:\n- ${latestUpload.source}`;
      if (onToken) {
        onToken(answer);
        return '';
      }
      return answer;
    }

    const vectorStore = await getVectorStore();
    if (!vectorStore) {
      answer = await runLlm(guidedQuestion);
    } else {
      const topK = parseInt(process.env.TOP_K_RETRIEVAL || '3');

      const docs = await withTimeout(
        hybridRetrieve(normalizedQuestion, topK),
        RETRIEVAL_TIMEOUT_MS,
        'Document retrieval timed out. Falling back to direct LLM answer.',
      ).catch((error) => {
        console.warn('RAG retrieval timeout/failure:', error?.message || error);
        return [];
      });

      if (docs.length > 0) {
        const context = buildContext(docs);
        answer = await runLlm(guidedQuestion, context);
        answer += buildSourcesFooter(docs);
      } else {
        if (ENABLE_DIRECT_LLM_FALLBACK) {
          // Out-of-RAG or retrieval timeout: answer with direct LLM (no retrieved context).
          answer = await runLlm(guidedQuestion);
        } else {
          answer = "I don't have enough grounded information to answer this question accurately.";
        }
      }
    }
  } catch (error: any) {
    console.error('RAG/LLM processing failure:', error?.message || error);
    answer = 'The answer service is currently slow. Please try again in a moment.';
  }

  if (onToken) {
    onToken(answer);
    return '';
  }

  return answer;
};

export default { processRagQuestion };
