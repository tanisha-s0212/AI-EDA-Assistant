export interface PredictionSchema {
  fields: { name: string; type: string; range?: { min: number; max: number } | null; categories: string[] }[];
  target: string;
  problem_type: string;
  trained_at?: string;
  dataset_version?: number;
  dataset_fingerprint?: string;
  readiness: { status: string; reasons: string[]; method?: string; rows?: number; metrics?: Record<string, number | null> };
}

export function predictionInputs(inputs: { name: string; role: string; value: string }[]): Record<string, string | number> {
  return Object.fromEntries(inputs.map(({ name, role, value }) => {
    if (!value.trim()) throw new Error(`Enter a value for "${name}".`);
    if (role === 'numeric') {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) throw new Error(`"${name}" must be a finite number.`);
      return [name, numeric];
    }
    return [name, value];
  }));
}

export function downloadPredictionCsv(filename: string, rows: unknown[][]): void {
  const escape = (value: unknown) => {
    let text = String(value ?? '');
    // Spreadsheet programs must treat user-provided values as text.
    if (/^[=+@\t\r]/.test(text) || /^-[^\d.]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  };
  const url = URL.createObjectURL(new Blob([rows.map(row => row.map(escape).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8;' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
