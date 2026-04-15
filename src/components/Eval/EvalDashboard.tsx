import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Download, Play, RefreshCw } from 'lucide-react';

type EvalSummary = {
  success: boolean;
  generated_at?: string;
  error?: string;
  summary?: {
    paper_count: number;
    by_stream: Record<string, { papers: number; avg_collaboration_score: number }>;
    by_agent: Record<string, number>;
    by_agent_detail?: Record<string, {
      agreement_to_manager: number;
      agreement_to_recommended: number | null;
      papers: number;
    }>;
    verdict_distribution?: Record<string, number>;
    error_category_distribution?: Record<string, number>;
    top_failures?: Array<{
      stream: string;
      paper_name: string;
      severity: number;
      error_categories: string[];
      issues: string[];
      recommended_final: string;
      question_path: string;
      collaboration_score: number;
    }>;
  };
  papers?: Array<{
    paper_index: number;
    stream: string;
    paper_name: string;
    scores: {
      collaboration_score: number;
      consensus_score: number;
      answer_overlap_score: number | null;
    };
    manager_review?: {
      verdict?: string;
      confidence?: number;
      issues?: string[];
    };
  }>;
};

type EvalJobStatus = {
  success: boolean;
  job_id?: string;
  status?: 'queued' | 'running' | 'completed' | 'failed';
  progress?: number;
  message?: string;
  result?: EvalSummary;
  error?: string;
};

const EvalDashboard: React.FC = () => {
  const [dseRoot, setDseRoot] = useState('/Users/SunnyMok/Desktop/DSE Math');
  const [includeAnswerScoring, setIncludeAnswerScoring] = useState(false);
  const [useVlmOcr, setUseVlmOcr] = useState(false);
  const [ocrMaxPages, setOcrMaxPages] = useState(3);
  const [maxPapers, setMaxPapers] = useState(5);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [error, setError] = useState('');
  const [jobId, setJobId] = useState('');
  const [jobStatus, setJobStatus] = useState('');
  const [jobProgress, setJobProgress] = useState(0);
  const [jobMessage, setJobMessage] = useState('');

  const loadSummary = async () => {
    setError('');
    try {
      const res = await fetch('/api/eval/summary');
      const data = await res.json();
      setSummary(data);
      if (!data?.success && data?.error) setError(data.error);
    } catch {
      setError('Failed to load evaluation summary.');
    }
  };

  useEffect(() => {
    void loadSummary();
  }, []);

  const runEval = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/eval/run-async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dse_root: dseRoot,
          include_answer_scoring: includeAnswerScoring,
          max_papers: maxPapers,
          use_vlm_ocr_for_scanned_pdf: useVlmOcr,
          ocr_max_pages: ocrMaxPages,
          split_train_ratio: trainRatio,
          split_val_ratio: valRatio,
          split_test_ratio: testRatio,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data?.success || !data?.job_id) {
        setError(data?.error || 'Failed to queue evaluation job.');
        return;
      }

      const id = String(data.job_id);
      setJobId(id);
      setJobStatus('queued');
      setJobProgress(0.01);
      setJobMessage('Queued...');

      while (true) {
        await new Promise((r) => setTimeout(r, 1500));
        const statusRes = await fetch(`/api/eval/job/${id}`);
        const statusData: EvalJobStatus = await statusRes.json();

        if (!statusRes.ok || !statusData?.success) {
          setError(statusData?.error || 'Failed to fetch job status.');
          break;
        }

        setJobStatus(String(statusData.status || 'running'));
        setJobProgress(Math.max(0, Math.min(1, Number(statusData.progress ?? 0))));
        setJobMessage(String(statusData.message || 'Running...'));

        if (statusData.status === 'completed') {
          if (statusData.result) {
            setSummary(statusData.result);
            if (!statusData.result.success) {
              setError(statusData.result.error || 'Evaluation completed with no results. Check folder structure/path.');
            }
          } else {
            await loadSummary();
          }
          break;
        }
        if (statusData.status === 'failed') {
          setError(statusData.error || 'Evaluation job failed.');
          break;
        }
      }
    } catch {
      setError('Failed to run evaluation.');
    } finally {
      setLoading(false);
    }
  };

  const runQuickEval = async () => {
    setMaxPapers(3);
    setUseVlmOcr(false);
    setIncludeAnswerScoring(false);
    setOcrMaxPages(1);

    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/eval/run-async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dse_root: dseRoot,
          include_answer_scoring: false,
          max_papers: 3,
          use_vlm_ocr_for_scanned_pdf: false,
          ocr_max_pages: 1,
          split_train_ratio: trainRatio,
          split_val_ratio: valRatio,
          split_test_ratio: testRatio,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data?.success || !data?.job_id) {
        setError(data?.error || 'Failed to queue quick evaluation job.');
        return;
      }

      const id = String(data.job_id);
      setJobId(id);
      setJobStatus('queued');
      setJobProgress(0.01);
      setJobMessage('Quick run queued...');

      while (true) {
        await new Promise((r) => setTimeout(r, 1200));
        const statusRes = await fetch(`/api/eval/job/${id}`);
        const statusData: EvalJobStatus = await statusRes.json();
        if (!statusRes.ok || !statusData?.success) {
          setError(statusData?.error || 'Failed to fetch job status.');
          break;
        }
        setJobStatus(String(statusData.status || 'running'));
        setJobProgress(Math.max(0, Math.min(1, Number(statusData.progress ?? 0))));
        setJobMessage(String(statusData.message || 'Running...'));
        if (statusData.status === 'completed') {
          if (statusData.result) setSummary(statusData.result);
          else await loadSummary();
          break;
        }
        if (statusData.status === 'failed') {
          setError(statusData.error || 'Evaluation job failed.');
          break;
        }
      }
    } catch {
      setError('Failed to run quick evaluation.');
    } finally {
      setLoading(false);
    }
  };

  const runOcrRetry = async () => {
    setUseVlmOcr(true);
    setOcrMaxPages(2);
    setMaxPapers(1);
    setIncludeAnswerScoring(false);

    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/eval/run-async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dse_root: dseRoot,
          include_answer_scoring: false,
          max_papers: 1,
          use_vlm_ocr_for_scanned_pdf: true,
          ocr_max_pages: 2,
          split_train_ratio: trainRatio,
          split_val_ratio: valRatio,
          split_test_ratio: testRatio,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data?.success || !data?.job_id) {
        setError(data?.error || 'Failed to queue OCR retry job.');
        return;
      }

      const id = String(data.job_id);
      setJobId(id);
      setJobStatus('queued');
      setJobProgress(0.01);
      setJobMessage('OCR retry queued...');

      while (true) {
        await new Promise((r) => setTimeout(r, 1500));
        const statusRes = await fetch(`/api/eval/job/${id}`);
        const statusData: EvalJobStatus = await statusRes.json();
        if (!statusRes.ok || !statusData?.success) {
          setError(statusData?.error || 'Failed to fetch OCR retry status.');
          break;
        }
        setJobStatus(String(statusData.status || 'running'));
        setJobProgress(Math.max(0, Math.min(1, Number(statusData.progress ?? 0))));
        setJobMessage(String(statusData.message || 'Running...'));
        if (statusData.status === 'completed') {
          if (statusData.result) {
            setSummary(statusData.result);
            if (!statusData.result.success && statusData.result.error) setError(statusData.result.error);
          } else {
            await loadSummary();
          }
          break;
        }
        if (statusData.status === 'failed') {
          setError(statusData.error || 'OCR retry failed.');
          break;
        }
      }
    } catch {
      setError('Failed to run OCR retry.');
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = async (type: 'papers' | 'failures') => {
    const res = await fetch(`/api/eval/report.csv?type=${type}`);
    if (!res.ok) {
      setError('Failed to download CSV report.');
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = type === 'papers' ? 'dse_eval_papers.csv' : 'dse_eval_failures.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const sortedPapers = useMemo(() => {
    return [...(summary?.papers || [])].sort(
      (a, b) => (b.scores?.collaboration_score || 0) - (a.scores?.collaboration_score || 0),
    );
  }, [summary]);

  const loraDecision = useMemo(() => {
    const paperCount = Number(summary?.summary?.paper_count || 0);
    const streamRows = Object.values(summary?.summary?.by_stream || {});
    const streamAvg = streamRows.length
      ? streamRows.reduce((acc, x) => acc + Number(x?.avg_collaboration_score || 0), 0) / streamRows.length
      : 0;
    const verdict = summary?.summary?.verdict_distribution || {};
    const pass = Number(verdict.pass || 0);
    const warn = Number(verdict.warn || 0);
    const fail = Number(verdict.fail || 0);
    const weakRatio = (warn + fail) / Math.max(1, pass + warn + fail);

    if (paperCount < 10) {
      return {
        label: 'Collect more eval data first',
        recommend: false,
        reason: 'Run at least 10 papers before deciding LoRA.',
      };
    }

    if (streamAvg < 0.7 || weakRatio > 0.45) {
      return {
        label: 'LoRA recommended',
        recommend: true,
        reason: `Average collaboration ${streamAvg.toFixed(3)}, warn/fail ratio ${(weakRatio * 100).toFixed(1)}%.`,
      };
    }

    return {
      label: 'LoRA optional',
      recommend: false,
      reason: `Average collaboration ${streamAvg.toFixed(3)}, warn/fail ratio ${(weakRatio * 100).toFixed(1)}%. Try prompt/retrieval tuning first.`,
    };
  }, [summary]);

  return (
    <div className="h-full overflow-y-auto p-6 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100">
      <div className="max-w-7xl mx-auto space-y-5">
        <div className="flex items-center gap-2">
          <BarChart3 className="text-blue-600" size={22} />
          <h2 className="text-xl font-bold">DSE Math Evaluation Dashboard</h2>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4 grid md:grid-cols-4 gap-3">
          <div className="md:col-span-2">
            <label className="text-xs text-slate-500">DSE folder path</label>
            <input
              value={dseRoot}
              onChange={(e) => setDseRoot(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
              placeholder="/Users/you/Desktop/DSE Math"
            />
          </div>

          <div>
            <label className="text-xs text-slate-500">Max papers</label>
            <input
              type="number"
              min={1}
              max={200}
              value={maxPapers}
              onChange={(e) => setMaxPapers(Number(e.target.value) || 20)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
            />
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={runEval}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 text-white px-3 py-2 hover:bg-blue-700 disabled:opacity-50"
            >
              <Play size={16} />
              {loading ? 'Running...' : 'Run evaluation'}
            </button>
            <button
              onClick={runQuickEval}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg border border-blue-300 dark:border-blue-700 px-3 py-2 disabled:opacity-50"
            >
              Quick run (3)
            </button>
            <button
              onClick={loadSummary}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2"
            >
              <RefreshCw size={16} />
              Reload
            </button>
          </div>

          <div className="md:col-span-4 flex flex-wrap gap-2">
            <button
              onClick={() => void downloadCsv('papers')}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 text-sm"
            >
              <Download size={14} />
              Download papers CSV
            </button>
            <button
              onClick={() => void downloadCsv('failures')}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 text-sm"
            >
              <Download size={14} />
              Download top-failures CSV
            </button>
          </div>

          <label className="md:col-span-4 text-sm flex items-center gap-2">
            <input
              type="checkbox"
              checked={includeAnswerScoring}
              onChange={(e) => setIncludeAnswerScoring(e.target.checked)}
            />
            Enable answer-key overlap scoring (answer files are used only after generation; no leakage into solving)
          </label>

          <label className="md:col-span-4 text-sm flex items-center gap-2">
            <input
              type="checkbox"
              checked={useVlmOcr}
              onChange={(e) => setUseVlmOcr(e.target.checked)}
            />
            Use VLM OCR fallback for scanned PDFs (pastpaper and answer extraction)
          </label>

          <div>
            <label className="text-xs text-slate-500">OCR max pages per PDF</label>
            <input
              type="number"
              min={1}
              max={12}
              value={ocrMaxPages}
              onChange={(e) => setOcrMaxPages(Number(e.target.value) || 3)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
            />
          </div>

          <div>
            <label className="text-xs text-slate-500">Train ratio</label>
            <input
              type="number"
              min={0.1}
              max={0.95}
              step={0.05}
              value={trainRatio}
              onChange={(e) => setTrainRatio(Number(e.target.value) || 0.8)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
            />
          </div>

          <div>
            <label className="text-xs text-slate-500">Val ratio</label>
            <input
              type="number"
              min={0.02}
              max={0.6}
              step={0.01}
              value={valRatio}
              onChange={(e) => setValRatio(Number(e.target.value) || 0.1)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
            />
          </div>

          <div>
            <label className="text-xs text-slate-500">Test ratio</label>
            <input
              type="number"
              min={0.02}
              max={0.6}
              step={0.01}
              value={testRatio}
              onChange={(e) => setTestRatio(Number(e.target.value) || 0.1)}
              className="mt-1 w-full rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 bg-white dark:bg-slate-900"
            />
          </div>
        </div>

        {loading && (
          <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/30 p-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span>Evaluation progress {jobStatus ? `(${jobStatus})` : ''}</span>
              <span>{Math.round(jobProgress * 100)}%</span>
            </div>
            <div className="h-2 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
              <div className="h-full bg-blue-600 transition-all" style={{ width: `${Math.max(2, Math.round(jobProgress * 100))}%` }} />
            </div>
            <div className="text-xs text-slate-600 dark:text-slate-300 mt-2">
              {jobMessage || 'Preparing...'} {jobId ? `• job ${jobId.slice(0, 8)}` : ''}
            </div>
          </div>
        )}

        {error && (
          <div className="text-sm text-red-500 flex flex-wrap items-center gap-3">
            <span>{error}</span>
            {error.toLowerCase().includes('no readable question text') && (
              <button
                onClick={runOcrRetry}
                disabled={loading}
                className="rounded-md border border-red-300 dark:border-red-700 px-2 py-1 text-xs"
              >
                Retry with OCR ON (1 paper)
              </button>
            )}
          </div>
        )}

        {!loading && !error && (summary?.summary?.paper_count || 0) === 0 && (
          <div className="rounded-xl border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/20 p-3 text-sm">
            No papers were evaluated yet. Check path structure (`core/m1/m2` with `pastpaper`) and scanned-PDF OCR availability.
          </div>
        )}

        <div className="grid md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4">
            <div className="text-xs text-slate-500">Papers evaluated</div>
            <div className="text-3xl font-semibold">{summary?.summary?.paper_count || 0}</div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4 md:col-span-2">
            <div className="text-xs text-slate-500 mb-2">Average collaboration by stream</div>
            <div className="grid grid-cols-3 gap-2 text-sm">
              {['CORE', 'M1', 'M2'].map((s) => (
                <div key={s} className="rounded-lg bg-slate-100 dark:bg-slate-800 p-2">
                  <div className="font-medium">{s}</div>
                  <div>
                    {summary?.summary?.by_stream?.[s]?.avg_collaboration_score ?? '-'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4">
          <div className="text-sm font-semibold mb-3">Agent agreement score</div>
          <div className="grid md:grid-cols-3 gap-2 text-sm">
            {Object.entries(summary?.summary?.by_agent || {}).map(([agent, score]) => (
              <div key={agent} className="rounded-lg bg-slate-100 dark:bg-slate-800 p-2 flex items-center justify-between">
                <span>{agent}</span>
                <span className="font-semibold">{Number(score).toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4 overflow-auto">
          <div className="text-sm font-semibold mb-3">Individual agent scorecard</div>
          <table className="w-full text-sm min-w-[680px]">
            <thead>
              <tr className="text-left border-b border-slate-200 dark:border-slate-700">
                <th className="py-2">Agent</th>
                <th className="py-2">Agreement to manager</th>
                <th className="py-2">Agreement to recommended</th>
                <th className="py-2">Papers</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary?.summary?.by_agent_detail || {}).map(([agent, row]) => (
                <tr key={agent} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="py-2 pr-2">{agent}</td>
                  <td className="py-2">{Number(row.agreement_to_manager || 0).toFixed(3)}</td>
                  <td className="py-2">{row.agreement_to_recommended == null ? '-' : Number(row.agreement_to_recommended).toFixed(3)}</td>
                  <td className="py-2">{row.papers}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className={`rounded-xl border p-4 ${loraDecision.recommend
          ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/20'
          : 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/20'}`}>
          <div className="text-sm font-semibold">LoRA decision helper</div>
          <div className="text-base mt-1">{loraDecision.label}</div>
          <div className="text-xs mt-1 text-slate-600 dark:text-slate-300">{loraDecision.reason}</div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4">
            <div className="text-sm font-semibold mb-3">Error category chart</div>
            <div className="space-y-2">
              {Object.entries(summary?.summary?.error_category_distribution || {}).map(([cat, count]) => {
                const max = Math.max(1, ...Object.values(summary?.summary?.error_category_distribution || { a: 1 }));
                const width = `${Math.round((Number(count) / max) * 100)}%`;
                return (
                  <div key={cat}>
                    <div className="flex justify-between text-xs">
                      <span>{cat}</span>
                      <span>{count}</span>
                    </div>
                    <div className="h-2 rounded bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full bg-amber-500" style={{ width }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4">
            <div className="text-sm font-semibold mb-3">Manager verdict distribution</div>
            <div className="space-y-2 text-sm">
              {Object.entries(summary?.summary?.verdict_distribution || {}).map(([k, v]) => (
                <div key={k} className="rounded-lg bg-slate-100 dark:bg-slate-800 p-2 flex items-center justify-between">
                  <span className="capitalize">{k}</span>
                  <span className="font-semibold">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4 overflow-auto">
          <div className="text-sm font-semibold mb-3 inline-flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-500" />
            Top failure cases for fine-tuning data mining
          </div>
          <table className="w-full text-sm min-w-[960px]">
            <thead>
              <tr className="text-left border-b border-slate-200 dark:border-slate-700">
                <th className="py-2">Paper</th>
                <th className="py-2">Stream</th>
                <th className="py-2">Severity</th>
                <th className="py-2">Collab</th>
                <th className="py-2">Categories</th>
                <th className="py-2">Issues</th>
                <th className="py-2">Recommended final</th>
              </tr>
            </thead>
            <tbody>
              {(summary?.summary?.top_failures || []).map((f, i) => (
                <tr key={`${f.paper_name}-${i}`} className="border-b border-slate-100 dark:border-slate-800 align-top">
                  <td className="py-2 pr-2">{f.paper_name}</td>
                  <td className="py-2">{f.stream}</td>
                  <td className="py-2">{Number(f.severity).toFixed(3)}</td>
                  <td className="py-2">{Number(f.collaboration_score || 0).toFixed(3)}</td>
                  <td className="py-2">{(f.error_categories || []).join(', ') || '-'}</td>
                  <td className="py-2">{(f.issues || []).join(' | ') || '-'}</td>
                  <td className="py-2 max-w-[300px] whitespace-pre-wrap">{f.recommended_final || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 p-4 overflow-auto">
          <div className="text-sm font-semibold mb-3">Per-paper results</div>
          <table className="w-full text-sm min-w-[780px]">
            <thead>
              <tr className="text-left border-b border-slate-200 dark:border-slate-700">
                <th className="py-2">Paper</th>
                <th className="py-2">Stream</th>
                <th className="py-2">Collab</th>
                <th className="py-2">Consensus</th>
                <th className="py-2">Answer overlap</th>
                <th className="py-2">Manager verdict</th>
              </tr>
            </thead>
            <tbody>
              {sortedPapers.map((p) => (
                <tr key={`${p.stream}-${p.paper_index}-${p.paper_name}`} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="py-2 pr-2">{p.paper_name}</td>
                  <td className="py-2">{p.stream}</td>
                  <td className="py-2">{p.scores?.collaboration_score?.toFixed?.(3) ?? p.scores?.collaboration_score}</td>
                  <td className="py-2">{p.scores?.consensus_score?.toFixed?.(3) ?? p.scores?.consensus_score}</td>
                  <td className="py-2">{p.scores?.answer_overlap_score == null ? '-' : Number(p.scores.answer_overlap_score).toFixed(3)}</td>
                  <td className="py-2">{p.manager_review?.verdict || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EvalDashboard;
