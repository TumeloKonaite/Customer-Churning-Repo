import { useRef, useState, type DragEvent } from "react";
import { Download, FileCheck2, FileSpreadsheet, RotateCcw, UploadCloud, X } from "lucide-react";
import { BatchResults } from "../components/BatchResults";
import { ApiErrorAlert, LoadingButton, PageHeader } from "../components/ui";
import { asApiError, predictBatch, type ApiError } from "../lib/api";
import { downloadText, SAMPLE_CSV, validateCsvFile } from "../lib/csv";
import type { BatchMode, BatchResponse } from "../types/api";

function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function BatchPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<BatchMode>("partial");
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const chooseFile = async (candidate?: File) => {
    setResult(null); setApiError(null);
    if (!candidate) return;
    const error = await validateCsvFile(candidate);
    setFileError(error);
    setFile(error ? null : candidate);
  };
  const drop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); void chooseFile(event.dataTransfer.files[0]); };
  const reset = () => { setFile(null); setFileError(null); setApiError(null); setResult(null); setMode("partial"); if (inputRef.current) inputRef.current.value = ""; };
  const process = async () => {
    if (!file) return;
    setLoading(true); setApiError(null); setResult(null);
    try { setResult(await predictBatch(file, mode)); }
    catch (error) { setApiError(asApiError(error)); }
    finally { setLoading(false); }
  };

  return <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
    <PageHeader eyebrow="Batch prediction" title="Assess a customer batch" description="Upload a CSV containing the model’s required columns. Each file can contain up to 100 customer records." />
    <div className="grid gap-6 lg:grid-cols-[1.2fr_.8fr]">
      <section className="card p-5 sm:p-7">
        <div className="flex items-center justify-between gap-3"><div><h2 className="text-lg font-bold text-navy-900">Customer CSV</h2><p className="mt-1 text-sm text-slate-600">CSV files only · maximum 100 records</p></div><button type="button" className="button-secondary min-h-10 px-3 py-2" onClick={() => downloadText("churn-insight-sample.csv", SAMPLE_CSV)}><Download className="h-4 w-4" /><span className="hidden sm:inline">Sample CSV</span></button></div>
        <div className={`mt-5 rounded-2xl border-2 border-dashed p-8 text-center transition ${dragging ? "border-teal-700 bg-teal-50" : fileError ? "border-red-300 bg-red-50" : "border-slate-300 bg-slate-50 hover:border-teal-600 hover:bg-teal-50/40"}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={drop}>
          <UploadCloud className="mx-auto h-9 w-9 text-teal-700" aria-hidden="true" /><p className="mt-3 font-semibold text-navy-900">Drag and drop your CSV here</p><p className="mt-1 text-sm text-slate-500">or use the standard file picker</p><input ref={inputRef} id="csv-file" type="file" accept=".csv,text/csv" className="sr-only" onChange={(event) => void chooseFile(event.target.files?.[0])} /><button type="button" className="button-secondary mt-4" onClick={() => inputRef.current?.click()}>Choose CSV file</button>
        </div>
        {fileError && <p className="mt-3 text-sm font-medium text-red-700" role="alert">{fileError}</p>}
        {file && <div className="mt-4 flex items-center gap-3 rounded-xl border border-teal-100 bg-teal-50 p-4"><FileCheck2 className="h-6 w-6 shrink-0 text-teal-800" /><div className="min-w-0 flex-1"><p className="truncate text-sm font-semibold text-navy-900">{file.name}</p><p className="text-xs text-slate-600">{fileSize(file.size)}</p></div><button type="button" onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }} className="rounded-lg p-2 text-slate-600 hover:bg-white" aria-label="Remove selected file"><X className="h-4 w-4" /></button></div>}
      </section>
      <section className="card p-5 sm:p-7">
        <h2 className="text-lg font-bold text-navy-900">Processing mode</h2><p className="mt-1 text-sm text-slate-600">Choose how validation errors should affect the batch.</p>
        <div className="mt-5 space-y-3">{([{ value: "partial", title: "Partial", copy: "Score valid rows and report invalid rows separately." }, { value: "fail_fast", title: "Fail fast", copy: "Stop processing when validation fails." }] as const).map((option) => <label key={option.value} className={`flex cursor-pointer gap-3 rounded-xl border p-4 transition focus-within:ring-2 focus-within:ring-teal-600 focus-within:ring-offset-2 ${mode === option.value ? "border-teal-700 bg-teal-50" : "border-slate-200 hover:bg-slate-50"}`}><input type="radio" name="mode" value={option.value} checked={mode === option.value} onChange={() => setMode(option.value)} className="mt-1 accent-teal-700" /><span><span className="block text-sm font-semibold text-navy-900">{option.title}</span><span className="mt-1 block text-xs leading-5 text-slate-600">{option.copy}</span></span></label>)}</div>
        <div className="mt-6 flex flex-col gap-3"><LoadingButton loading={loading} onClick={process} disabled={!file}><FileSpreadsheet className="h-4 w-4" /> Process batch</LoadingButton><button type="button" className="button-secondary" onClick={reset}><RotateCcw className="h-4 w-4" /> Clear all</button></div>
        {loading && <div className="mt-5" role="status" aria-live="polite"><div className="h-2 overflow-hidden rounded-full bg-slate-200"><div className="h-full w-2/3 animate-pulse rounded-full bg-teal-700 motion-reduce:animate-none" /></div><p className="mt-2 text-xs text-slate-600">Uploading and processing the file…</p></div>}
      </section>
    </div>
    {apiError && <div className="mt-6"><ApiErrorAlert error={apiError} /></div>}
    {result && <div className="mt-8"><BatchResults response={result} /></div>}
    {!result && !apiError && !loading && <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-8 text-center"><FileSpreadsheet className="mx-auto h-8 w-8 text-slate-400" /><h2 className="mt-3 font-semibold text-navy-900">Results will appear here</h2><p className="mt-1 text-sm text-slate-500">Select a valid CSV and process the batch to review model estimates.</p></div>}
  </div>;
}
