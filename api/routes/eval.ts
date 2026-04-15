import { Router, type Request, type Response } from 'express';
import axios from 'axios';

const router = Router();
const pyBase = (process.env.PY_BACKEND_URL || 'http://localhost:8002').replace(/\/$/, '');

async function proxyJson(req: Request, res: Response, method: 'GET' | 'POST', targetPath: string) {
  try {
    const url = `${pyBase}${targetPath}`;
    const result = await axios({
      method,
      url,
      data: req.body,
      params: req.query,
      timeout: 1000 * 60 * 30,
      responseType: 'json',
    });
    res.status(result.status).json(result.data);
  } catch (error: any) {
    const code = error?.response?.status || 502;
    const msg = error?.response?.data || error?.message || 'Failed to proxy to python backend';
    res.status(code).json({ success: false, error: msg });
  }
}

router.post('/run', async (req, res) => proxyJson(req, res, 'POST', '/api/eval/run'));
router.post('/run-async', async (req, res) => proxyJson(req, res, 'POST', '/api/eval/run-async'));
router.get('/summary', async (req, res) => proxyJson(req, res, 'GET', '/api/eval/summary'));
router.get('/job/:job_id', async (req, res) => {
  try {
    const url = `${pyBase}/api/eval/job/${encodeURIComponent(req.params.job_id)}`;
    const result = await axios.get(url, { timeout: 1000 * 60 * 30, responseType: 'json' });
    res.status(result.status).json(result.data);
  } catch (error: any) {
    const code = error?.response?.status || 502;
    const msg = error?.response?.data || error?.message || 'Failed to proxy to python backend';
    res.status(code).json({ success: false, error: msg });
  }
});

router.get('/report.csv', async (req, res) => {
  try {
    const url = `${pyBase}/api/eval/report.csv`;
    const result = await axios.get(url, {
      params: req.query,
      timeout: 1000 * 60 * 30,
      responseType: 'arraybuffer',
      headers: { Accept: 'text/csv' },
    });

    const cd = result.headers['content-disposition'] || 'attachment; filename="dse_eval.csv"';
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', cd);
    res.status(result.status).send(Buffer.from(result.data));
  } catch (error: any) {
    const code = error?.response?.status || 502;
    const msg = error?.response?.data || error?.message || 'Failed to proxy CSV from python backend';
    res.status(code).json({ success: false, error: msg });
  }
});

export default router;
