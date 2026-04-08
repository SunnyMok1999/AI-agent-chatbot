import { Request, Response } from 'express';
import { processRagQuestion } from '../services/rag.service.js';
import { supabase } from '../services/supabase.service.js';

export const handleChat = async (req: Request, res: Response) => {
  const { message, conversation_id, stream = true } = req.body;

  if (!message) {
    return res.status(400).json({ success: false, error: 'Message is required' });
  }

  // Handle streaming response
  if (stream) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
      await processRagQuestion(message, (token) => {
        res.write(`data: ${JSON.stringify({ type: 'chunk', content: token })}\n\n`);
      });
      res.write(`data: ${JSON.stringify({ type: 'complete' })}\n\n`);
      res.end();
    } catch (error: any) {
      console.error('Error in chat processing:', error);
      res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
      res.end();
    }
  } else {
    // Non-streaming response
    try {
      const response = await processRagQuestion(message);
      res.status(200).json({ success: true, content: response });
    } catch (error: any) {
      console.error('Error in chat processing:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  }
};

export default { handleChat };
