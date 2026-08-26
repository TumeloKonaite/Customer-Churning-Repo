import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import type { BatchResponse } from "../types/api";
import { MetadataDetails } from "./ui";

export function BatchResults({ response }: { response: BatchResponse }) {
  const [filter, setFilter] = useState("");
  const results = useMemo(() => {
    const term = filter.trim().toLowerCase();
    if (!term) return response.results;
    return response.results.filter((result) => String(result.id ?? result.index + 1).toLowerCase().includes(term) || (result.predicted_label === 1 ? "higher" : "lower").includes(term));
  }, [filter, response.results]);
  const isPartial = response.status === "partial" || (response.summary.invalid_records > 0 && response.summary.valid_records > 0);
  const isFailed = response.summary.valid_records === 0 || response.status === "failed" || response.status === "error";
  const title = isFailed ? "Batch could not be scored" : isPartial ? "Batch completed with some errors" : "Batch completed successfully";

  return <section className="space-y-6" aria-live="polite">
    <div className={`rounded-2xl border p-5 ${isFailed ? "border-red-200 bg-red-50" : isPartial ? "border-amber-200 bg-amber-50" : "border-teal-100 bg-teal-50"}`}>
      <h2 className="text-lg font-bold text-navy-950">{title}</h2><p className="mt-1 text-sm text-slate-600">Review the summary and row-level details below. All predictions remain model estimates.</p>
    </div>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {[["Total rows", response.summary.total_records], ["Predictions", response.summary.valid_records], ["Failed rows", response.summary.invalid_records], ["Mode", response.summary.mode === "fail_fast" ? "Fail fast" : "Partial"]].map(([label, value]) => <div className="card p-5" key={label}><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p><p className="mt-2 text-2xl font-bold text-navy-950">{value}</p></div>)}
    </div>
    {response.results.length > 0 && <div className="card overflow-hidden">
      <div className="flex flex-col gap-4 border-b p-5 sm:flex-row sm:items-center sm:justify-between"><div><h2 className="font-bold text-navy-900">Prediction results</h2><p className="mt-1 text-sm text-slate-500">Showing {results.length} of {response.results.length} scored rows</p></div><label className="relative"><span className="sr-only">Filter results</span><Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-slate-400" /><input className="field py-2 pl-9 text-sm" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter by ID or risk" /></label></div>
      <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-sm"><thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-5 py-3">Row</th><th className="px-5 py-3">Customer ID</th><th className="px-5 py-3">Risk estimate</th><th className="px-5 py-3">Probability</th></tr></thead><tbody className="divide-y divide-slate-100">{results.map((result) => <tr key={`${result.index}-${String(result.id)}`}><td className="px-5 py-4 font-medium">{result.index + 1}</td><td className="px-5 py-4 text-slate-600">{result.id == null ? "Not provided" : String(result.id)}</td><td className="px-5 py-4"><span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${result.predicted_label === 1 ? "bg-amber-50 text-amber-800" : "bg-teal-50 text-teal-800"}`}>{result.predicted_label === 1 ? "Higher churn risk" : "Lower churn risk"}</span></td><td className="px-5 py-4 font-medium">{result.p_churn == null ? "Unavailable" : `${(result.p_churn * 100).toFixed(1)}%`}</td></tr>)}</tbody></table>{!results.length && <p className="p-8 text-center text-sm text-slate-500">No results match that filter.</p>}</div>
    </div>}
    {response.errors?.length ? <div className="card overflow-hidden"><div className="border-b p-5"><h2 className="font-bold text-navy-900">Rows needing attention</h2><p className="mt-1 text-sm text-slate-500">These records were not scored.</p></div><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-sm"><thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-5 py-3">Row</th><th className="px-5 py-3">Customer ID</th><th className="px-5 py-3">Field</th><th className="px-5 py-3">Issue</th></tr></thead><tbody className="divide-y divide-slate-100">{response.errors.map((error, index) => <tr key={`${error.row_index}-${index}`}><td className="px-5 py-4 font-medium">{error.row_index + 1}</td><td className="px-5 py-4 text-slate-600">{error.id == null ? "Not provided" : String(error.id)}</td><td className="px-5 py-4">{error.field ?? "—"}</td><td className="px-5 py-4 text-red-800">{error.message}</td></tr>)}</tbody></table></div></div> : null}
    <MetadataDetails entries={[["Model", response.metadata.model_name], ["Model version", response.metadata.model_version], ["Model version ID", response.metadata.model_version_id], ["Deployment ID", response.metadata.deployment_id], ["MLflow run", response.metadata.mlflow_run_id], ["Processed at", response.timestamp]]} />
  </section>;
}
