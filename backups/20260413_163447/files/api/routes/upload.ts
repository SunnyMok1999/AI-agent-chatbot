import { Router } from 'express';
import multer from 'multer';
import { handleUpload } from '../controllers/upload.controller.js';

const router = Router();

const allowedMimeTypes = new Set([
  'application/pdf',
  'text/plain',
  'text/markdown',
  'image/png',
  'image/x-png',
]);

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, process.env.UPLOAD_DIRECTORY || './uploads');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760'), // 10MB
  },
  fileFilter: (_req, file, cb) => {
    if (!allowedMimeTypes.has(file.mimetype)) {
      cb(new Error('Unsupported file format. Please upload PDF, TXT, MD, or PNG.'));
      return;
    }
    cb(null, true);
  },
});

router.post('/', upload.single('file'), handleUpload);

export default router;
