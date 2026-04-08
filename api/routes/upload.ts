import { Router } from 'express';
import multer from 'multer';
import { handleUpload } from '../controllers/upload.controller.js';
import path from 'path';

const router = Router();

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
});

router.post('/', upload.single('file'), handleUpload);

export default router;
