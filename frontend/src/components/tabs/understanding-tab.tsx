'use client';

import { motion } from 'framer-motion';
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
  const numericColumns = columns.filter((col) => col.role === 'numeric');
  const categoricalColumns = columns.filter((col) => col.role === 'categorical' || col.role === 'boolean');
  const datetimeColumns = columns.filter((col) => col.role === 'datetime');
  const identifierColumns = columns.filter((col) => col.role === 'identifier');
  const highlyUniqueColumns = columns.filter((col) => totalRows > 0 && col.uniqueCount / totalRows >= 0.9);
  const sparseColumns = [...missingColumns].sort((a, b) => b.nullCount - a.nullCount).slice(0, 3);
  const completeness = totalRows && columns.length
    ? Math.round((columns.reduce((sum, col) => sum + col.nonNull, 0) / (totalRows * columns.length)) * 1000) / 10
    : 0;
  const qualitySignals = [
    `${completeness}% overall completeness across ${columns.length} columns`,
    duplicates > 0 ? `${duplicates.toLocaleString()} duplicate rows may need review` : 'No duplicate rows detected in the uploaded dataset',
    missingColumns.length > 0 ? `${missingColumns.length} columns contain missing values` : 'No columns with missing values detected',
  ];
  const structureSignals = [
    `${numericColumns.length} numeric column${numericColumns.length === 1 ? '' : 's'} for measures and trends`,
    `${categoricalColumns.length} categorical/boolean column${categoricalColumns.length === 1 ? '' : 's'} for segments and classes`,
    datetimeColumns.length > 0 ? `${datetimeColumns.length} datetime column${datetimeColumns.length === 1 ? '' : 's'} can support time-based analysis` : 'No datetime columns detected',
  ];
  const modelingSignals = [
    identifierColumns.length > 0 ? `${identifierColumns.length} identifier-like column${identifierColumns.length === 1 ? '' : 's'} should usually be excluded from modeling` : 'No strong identifier columns detected',
    highlyUniqueColumns.length > 0 ? `${highlyUniqueColumns.length} high-cardinality column${highlyUniqueColumns.length === 1 ? '' : 's'} may require encoding care` : 'Column cardinality looks manageable for standard ML workflows',
    sparseColumns.length > 0 ? `Most incomplete column: ${sparseColumns[0].name} (${Math.round((sparseColumns[0].nullCount / Math.max(totalRows, 1)) * 100)}% missing)` : 'Missingness risk is low across the current dataset view',
  ];

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
        <p className="mt-1 text-muted-foreground">Review the uploaded dataset identity, quality checks, explainability signals, and a quick preview before exploratory data analysis and data cleaning.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {[
          { label: 'Dataset Name', value: fileName ?? 'Uploaded dataset', Icon: FileText },
          { label: 'Total Records', value: totalRows.toLocaleString(), Icon: Database },
          { label: 'Total Features', value: String(columns.length), Icon: TableIcon },
          { label: 'Duplicates Found', value: duplicates.toLocaleString(), Icon: Sparkles },
          { label: 'Memory Estimate', value: memoryUsage || 'N/A', Icon: Database },
        ].map(({ label, value, Icon }) => (
          <motion.div
            key={label}
            whileHover={{ y: -8, rotateX: 3, rotateY: -3 }}
            transition={{ type: 'spring', stiffness: 260, damping: 18 }}
            className="[transform-style:preserve-3d]"
          >
          <Card className="group relative overflow-hidden border-border/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.72),rgba(248,250,252,0.94))] transition-all duration-500 hover:border-primary/30 hover:shadow-[0_28px_65px_-32px_rgba(37,99,235,0.38)] dark:bg-[linear-gradient(180deg,rgba(30,41,59,0.7),rgba(15,23,42,0.92))]">
            <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
              <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary/70 via-sky-400/65 to-emerald-400/65" />
              <div className="absolute -right-10 top-0 h-24 w-24 rounded-full bg-sky-400/15 blur-2xl" />
              <div className="absolute -left-8 bottom-0 h-20 w-20 rounded-full bg-emerald-400/10 blur-2xl" />
            </div>
            <CardContent className="relative flex items-center gap-3 pt-6">
              <div className="rounded-2xl bg-primary/10 p-3 transition-all duration-500 group-hover:scale-110 group-hover:bg-primary/15 group-hover:shadow-[0_14px_30px_-18px_rgba(37,99,235,0.55)]"><Icon className="h-5 w-5 text-primary transition-transform duration-500 group-hover:rotate-6" /></div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground transition-colors duration-300 group-hover:text-foreground/70">{label}</p>
                <p className="truncate text-lg font-semibold transition-transform duration-500 group-hover:translate-x-0.5">{value}</p>
              </div>
            </CardContent>
          </Card>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-primary/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> AI Explainability</CardTitle>
            <CardDescription>What the uploaded dataset is telling us about quality and readiness.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {qualitySignals.map((signal) => (
              <div key={signal} className="rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                <p className="transition-colors duration-300 group-hover:text-foreground/85">{signal}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-sky-400/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><Database className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> Dataset Structure</CardTitle>
            <CardDescription>How the uploaded dataset is organized for analysis.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {structureSignals.map((signal) => (
              <div key={signal} className="rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                <p className="transition-colors duration-300 group-hover:text-foreground/85">{signal}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-emerald-400/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><TableIcon className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> Modeling Readiness</CardTitle>
            <CardDescription>Key explainability cues that can affect downstream ML behavior.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {modelingSignals.map((signal) => (
              <div key={signal} className="rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                <p className="transition-colors duration-300 group-hover:text-foreground/85">{signal}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>


      <Card className="group overflow-hidden border-border/70 bg-card/80 shadow-[0_18px_48px_-36px_rgba(15,23,42,0.22)] transition-all duration-500 hover:border-primary/30 hover:shadow-[0_28px_65px_-34px_rgba(37,99,235,0.3)]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4 text-primary transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3" /> Data Preview</CardTitle>
          <CardDescription>First {Math.min(10, data.length)} rows from the current dataset view.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-[420px] overflow-auto rounded-2xl border border-border/70 bg-background/70 shadow-inner transition-all duration-500 group-hover:border-primary/20">
            <ShadTable>
              <TableHeader>
                <TableRow>
                  <TableHead>SNo</TableHead>
                  {columns.map((col) => <TableHead key={col.name}>{col.name}</TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewRows.map((row, rowIdx) => (
                  <TableRow key={rowIdx} className="transition-all duration-300 hover:bg-primary/5 hover:shadow-[inset_3px_0_0_rgba(37,99,235,0.8)]">
                    <TableCell className="font-medium text-muted-foreground transition-colors duration-300 group-hover:text-foreground/80">{rowIdx + 1}</TableCell>
                    {columns.map((col) => <TableCell key={`${rowIdx}-${col.name}`} className="transition-colors duration-300 hover:text-foreground">{row[col.name] == null ? '-' : String(row[col.name])}</TableCell>)}
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
