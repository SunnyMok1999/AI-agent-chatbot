/**
 * This is a API server
 */

import express, {
  type Request,
  type Response,
  type NextFunction,
} from 'express'
import cors from 'cors'
import multer from 'multer'
import path from 'path'
import dotenv from 'dotenv'
import { fileURLToPath } from 'url'
import authRoutes from './routes/auth.js'
import chatRoutes from './routes/chat.js'
import uploadRoutes from './routes/upload.js'
import evalRoutes from './routes/eval.js'

// for esm mode
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// load env
dotenv.config()

const app: express.Application = express()
const enablePublicUpload = process.env.ENABLE_PUBLIC_UPLOAD === 'true'

app.use(cors())
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true, limit: '10mb' }))

/**
 * API Routes
 */
app.use('/api/auth', authRoutes)
app.use('/api/chat', chatRoutes)
app.use('/api/eval', evalRoutes)
if (enablePublicUpload) {
  app.use('/api/upload', uploadRoutes)
} else {
  app.use('/api/upload', (_req: Request, res: Response) => {
    res.status(403).json({
      success: false,
      error: 'Public upload is disabled. Use backend ingestion script instead.',
    })
  })
}

/**
 * health
 */
app.use(
  '/api/health',
  (req: Request, res: Response, next: NextFunction): void => {
    res.status(200).json({
      success: true,
      message: 'ok',
    })
  },
)

/**
 * error handler middleware
 */
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      const maxFileSizeBytes = parseInt(process.env.MAX_FILE_SIZE || '10485760')
      const maxFileSizeMB = Math.round((maxFileSizeBytes / 1024 / 1024) * 10) / 10
      return res.status(413).json({
        success: false,
        error: `File is too large. Maximum allowed size is ${maxFileSizeMB}MB.`,
      })
    }

    return res.status(400).json({
      success: false,
      error: error.message,
    })
  }

  res.status(500).json({
    success: false,
    error: error.message || 'Server internal error',
  })
})

/**
 * 404 handler
 */
app.use((req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: 'API not found',
  })
})

export default app
