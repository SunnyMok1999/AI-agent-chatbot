import { Request, Response } from 'express';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { createVectorStore } from '../services/vectorstore.service.js';
import path from 'path';
import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

const uploadDir = process.env.UPLOAD_DIRECTORY || './uploads';

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
    let loader;
    if (mimetype === 'application/pdf') {
      loader = new PDFLoader(filePath);
    } else if (mimetype === 'text/plain' || mimetype === 'text/markdown') {
      loader = new TextLoader(filePath);
    } else {
      // Cleanup file if format not supported
      fs.unlinkSync(filePath);
      return res.status(400).json({ success: false, error: 'Unsupported file format' });
    }

    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: parseInt(process.env.CHUNK_SIZE || '1000'),
      chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '200'),
    });

    const chunks = await splitter.splitDocuments(docs);

    // Ingest into vector store
    await createVectorStore(chunks);

    // Cleanup uploaded file after processing
    fs.unlinkSync(filePath);

    return res.status(200).json({
      success: true,
      message: 'Document processed and ingested successfully',
      chunks_count: chunks.length,
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
