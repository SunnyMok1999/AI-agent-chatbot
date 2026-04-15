import { Request, Response } from 'express';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { Document } from '@langchain/core/documents';
import { createVectorStore } from '../services/vectorstore.service.js';
import {
  setLatestUploadContext,
  setLatestUploadImageContext,
} from '../services/upload-context.service.js';
import path from 'path';
import fs from 'fs';
import dotenv from 'dotenv';
import Tesseract from 'tesseract.js';

dotenv.config();

const uploadDir = process.env.UPLOAD_DIRECTORY || './uploads';
const ENABLE_TESSERACT_OCR = process.env.ENABLE_TESSERACT_OCR === 'true';
const ENABLE_PNG_OCR_INGEST = process.env.ENABLE_PNG_OCR_INGEST === 'true';
let ocrPipelinePromise: Promise<any> | null = null;

async function getOcrPipeline() {
  if (!ocrPipelinePromise) {
    ocrPipelinePromise = (async () => {
      const { pipeline, env } = await import('@xenova/transformers');
      env.allowLocalModels = true;
      env.useBrowserCache = false;
      return await pipeline('image-to-text', process.env.OCR_MODEL || 'Xenova/trocr-base-printed');
    })();
  }

  return ocrPipelinePromise;
}

async function extractTextFromPng(filePath: string): Promise<string> {
  try {
    const ocr = await getOcrPipeline();
    const result = await ocr(filePath);

    if (Array.isArray(result)) {
      return result.map((item: any) => item?.generated_text || '').join('\n').trim();
    }

    return String(result?.generated_text || '').trim();
  } catch (error) {
    console.warn('Primary OCR failed for PNG, trying Tesseract fallback...', error);
    return '';
  }
}

async function extractTextFromPngWithTesseract(filePath: string): Promise<string> {
  try {
    const result = await Tesseract.recognize(filePath, 'eng');
    return String(result?.data?.text || '').trim();
  } catch (error) {
    console.warn('Tesseract OCR fallback failed for PNG.', error);
    return '';
  }
}

function normalizeExtractedText(text: string): string {
  return text.replace(/\r/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

// Ensure upload directory exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

export const handleUpload = async (req: Request, res: Response) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded' });
  }

  const { originalname, path: filePath, mimetype } = req.file;

  try {
    let docs: Document[];
    let imageOnlyUpload = false;

    if (mimetype === 'application/pdf') {
      const loader = new PDFLoader(filePath);
      docs = await loader.load();
    } else if (mimetype === 'text/plain' || mimetype === 'text/markdown') {
      const loader = new TextLoader(filePath);
      docs = await loader.load();
    } else if (mimetype === 'image/png') {
      const imageBuffer = fs.readFileSync(filePath);
      const dataUrl = `data:${mimetype};base64,${imageBuffer.toString('base64')}`;
      setLatestUploadImageContext(originalname, mimetype, dataUrl);

      if (!ENABLE_PNG_OCR_INGEST) {
        imageOnlyUpload = true;
        docs = [];
      } else {
      let extractedText = normalizeExtractedText(await extractTextFromPng(filePath));

      // Fallback for cases where the primary OCR model returns too little/empty text.
      if (ENABLE_TESSERACT_OCR && (!extractedText || extractedText.replace(/\s+/g, '').length < 8)) {
        extractedText = normalizeExtractedText(await extractTextFromPngWithTesseract(filePath));
      }

      if (!extractedText) {
        throw new Error(
          'Could not extract text from PNG image. Please use a clear screenshot with readable text.',
        );
      }

      docs = [
        new Document({
          pageContent: extractedText,
          metadata: {
            source: originalname,
            file_type: 'image/png',
          },
        }),
      ];
      }
    } else {
      // Cleanup file if format not supported
      fs.unlinkSync(filePath);
      return res.status(400).json({ success: false, error: 'Unsupported file format' });
    }

    if (imageOnlyUpload) {
      fs.unlinkSync(filePath);
      return res.status(200).json({
        success: true,
        message: 'PNG uploaded for direct vision-model question answering',
        chunks_count: 0,
        preview: '[image stored for VLM] ',
      });
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: parseInt(process.env.CHUNK_SIZE || '1000'),
      chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '200'),
    });

    const chunks = await splitter.splitDocuments(docs);

    // Ingest into vector store
    await createVectorStore(chunks);

    // Keep recent upload context in memory for follow-up questions like
    // "answer the question in the uploaded image".
    const latestContent = chunks
      .slice(0, 3)
      .map((c) => c.pageContent || '')
      .join('\n')
      .trim();
    if (latestContent) {
      setLatestUploadContext(originalname, latestContent);
    }

    // Cleanup uploaded file after processing
    fs.unlinkSync(filePath);

    return res.status(200).json({
      success: true,
      message: 'Document processed and ingested successfully',
      chunks_count: chunks.length,
      preview: String(chunks[0]?.pageContent || '').slice(0, 220),
    });
  } catch (error: any) {
    console.error('Error processing document:', error);
    // Cleanup file in case of error
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    return res.status(500).json({ success: false, error: error.message });
  }
};

export default { handleUpload };
