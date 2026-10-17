'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { clearStaleModelState } from '@/lib/model-lifecycle';
import { downloadPredictionCsv } from '@/lib/prediction-contract';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from '@/components/ui/table';

interface BatchRow {
  row: number;
  status: string;
  prediction?: number | string;
  prediction_label?: number | string;
  error?: string;
  warnings?: string[];
  confidence?: number;
  inputs?: Record<string, unknown>;
  model_id?: string;
  dataset_version?: number;
  dataset_fingerprint?: string;
  scored_at?: string;
  error_band?: { lower: number; upper: number };
  readiness?: { status: string; reasons: string[] };
  probability_note?: string;
}

export function PredictionBatch({ modelId, datasetId }: { modelId: string; datasetId: string | null }) {
  const picker = useRef<HTMLInputElement>(null);
  const active = useRef(true);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const [sheet, setSheet] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<BatchRow[]>([]);
  const [summary, setSummary] = useState<{ total: number; succeeded: number; failed: number; with_warnings: number } | null>(null);
  const upload = async (file?: File) => {
    if (!file) return;
    setError(''); setResults([]); setSummary(null);
    if (file.size > 50 * 1024 * 1024) { setError('Use a file of 50 MB or smaller.'); return; }
    setBusy(true);
    try {
      const body = new FormData();
      body.append('file', file); body.append('model_id', modelId);
      if (datasetId) body.append('dataset_id', datasetId);
      if (sheet.trim()) body.append('sheet_name', sheet.trim());
      const response = await apiClient.post('/predict/batch', body);
      if (!active.current) return;
      setResults(response.data.results); setSummary(response.data.summary);
    } catch (failure) {
      if (!active.current) return;
      clearStaleModelState(failure);
      setError(getApiErrorMessage(failure, 'Batch prediction failed.'));
    } finally { if (active.current) setBusy(false); }
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Batch Prediction</CardTitle>
        <CardDescription>CSV, TSV, Excel, or Parquet. Use model feature names as headers; every data row receives a result or an error. Limit: 5,000 rows and 50 MB.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="space-y-1 text-xs">Excel worksheet name (required for workbooks with multiple sheets)
            <Input value={sheet} onChange={event => setSheet(event.target.value)} placeholder="Worksheet name" disabled={busy} />
          </label>
          <input ref={picker} type="file" accept=".csv,.tsv,.xlsx,.xls,.parquet" hidden onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; void upload(file); }} />
          <Button onClick={() => picker.current?.click()} disabled={busy}>{busy ? 'Scoring rows…' : 'Upload prediction inputs'}</Button>
          {results.length > 0 && <Button variant="outline" onClick={() => downloadPredictionCsv('prediction_batch.csv', [
            ['Data row', 'Status', 'Prediction', 'Model probability (uncalibrated)', 'Error', 'Warnings', 'Inputs', 'Model ID', 'Dataset version', 'Dataset fingerprint', 'Scored at', 'Lower approximate bound', 'Upper approximate bound', 'Readiness', 'Readiness reasons', 'Probability note'],
            ...results.map(row => [row.row, row.status, row.prediction_label ?? row.prediction, row.confidence, row.error, row.warnings?.join(' '), JSON.stringify(row.inputs), row.model_id, row.dataset_version, row.dataset_fingerprint, row.scored_at, row.error_band?.lower, row.error_band?.upper, row.readiness?.status, row.readiness?.reasons.join(' '), row.probability_note]),
          ])}>Export all results and errors</Button>}
        </div>
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        {summary && <p className="text-sm">{summary.total} rows: {summary.succeeded} succeeded, {summary.failed} rejected, {summary.with_warnings} with input warnings. Row numbers count data rows after the header.</p>}
        {results.length > 0 && <div className="max-h-80 overflow-auto"><Table>
          <TableHeader><TableRow><TableHead>Data row</TableHead><TableHead>Status</TableHead><TableHead>Prediction</TableHead><TableHead>Review details</TableHead></TableRow></TableHeader>
          <TableBody>{results.map(row => <TableRow key={row.row}>
            <TableCell>{row.row}</TableCell><TableCell>{row.status}</TableCell>
            <TableCell>{row.prediction_label ?? row.prediction ?? '—'}</TableCell>
            <TableCell>{row.error || row.warnings?.join(' ') || '—'}</TableCell>
          </TableRow>)}</TableBody>
        </Table></div>}
      </CardContent>
    </Card>
  );
}
