'use client';

import { AlertCircle, Database, Eye, FileText, Sparkles, Table as TableIcon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table as ShadTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useAppStore } from '@/lib/store';

export default function UnderstandingTab() {
  const rawData = useAppStore((s) => s.rawData);
  const cleanedData = useAppStore((s) => s.cleanedData);
  const columns = useAppStore((s) => s.columns);
  const totalRows = useAppStore((s) => s.totalRows);
  const duplicates = useAppStore((s) => s.duplicates);
  const memoryUsage = useAppStore((s) => s.memoryUsage);
  const fileName = useAppStore((s) => s.fileName);

  const data = cleanedData ?? rawData ?? [];
  const hasData = !!rawData && columns.length > 0;
  const previewRows = data.slice(0, 10);
  const missingColumns = columns.filter((col) => col.nullCount > 0);
  const completeness = totalRows && columns.length
    ? Math.round((columns.reduce((sum, col) => sum + col.nonNull, 0) / (totalRows * columns.length)) * 1000) / 10
    : 0;

  if (!hasData) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Data Understanding</h2>
          <p className="mt-1 text-muted-foreground">Upload a dataset to view data quality and preview details here.</p>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <p className="font-medium">No dataset loaded</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Data Understanding</h2>
        <p className="mt-1 text-muted-foreground">Review the uploaded dataset identity, quality checks, and a quick preview before cleaning and EDA.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="secondary" className="gap-1"><FileText className="h-3.5 w-3.5" /> {fileName ?? 'Uploaded dataset'}</Badge>
          <Badge variant="secondary" className="gap-1"><Database className="h-3.5 w-3.5" /> {data.length.toLocaleString()} rows</Badge>
          <Badge variant="secondary" className="gap-1"><TableIcon className="h-3.5 w-3.5" /> {columns.length} columns</Badge>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {[
          { label: 'Dataset Name', value: fileName ?? 'Uploaded dataset', Icon: FileText },
          { label: 'Total Records', value: totalRows.toLocaleString(), Icon: Database },
          { label: 'Total Features', value: String(columns.length), Icon: TableIcon },
          { label: 'Duplicates Found', value: duplicates.toLocaleString(), Icon: Sparkles },
          { label: 'Memory Estimate', value: memoryUsage || 'N/A', Icon: Database },
        ].map(({ label, value, Icon }) => (
          <Card key={label}>
            <CardContent className="flex items-center gap-3 pt-6">
              <div className="rounded-xl bg-emerald-500/10 p-3"><Icon className="h-5 w-5 text-emerald-600" /></div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">{label}</p>
                <p className="truncate text-lg font-semibold">{value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>


      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4 text-teal-600" /> Data Preview</CardTitle>
          <CardDescription>First {Math.min(10, data.length)} rows from the current dataset view.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-[420px] overflow-auto rounded-lg border">
            <ShadTable>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  {columns.map((col) => <TableHead key={col.name}>{col.name}</TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewRows.map((row, rowIdx) => (
                  <TableRow key={rowIdx}>
                    <TableCell>{rowIdx + 1}</TableCell>
                    {columns.map((col) => <TableCell key={`${rowIdx}-${col.name}`}>{row[col.name] == null ? '-' : String(row[col.name])}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </ShadTable>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
