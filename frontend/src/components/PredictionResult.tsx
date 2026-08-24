import { AlertTriangle, CheckCircle2, RotateCcw, SlidersHorizontal } from "lucide-react";
import type { PredictionResponse } from "../types/api";
import { MetadataDetails } from "./ui";

export function PredictionResult({ result, onAgain, onEdit }: { result: PredictionResponse; onAgain: () => void; onEdit: () => void }) {
  const highRisk = result.predicted_label === 1;
  const probability = result.p_churn == null ? null : Math.max(0, Math.min(1, result.p_churn));
  return <section className="card overflow-hidden" aria-live="polite">
    <div className={`border-b p-6 sm:p-8 ${highRisk ? "border-amber-200 bg-amber-50" : "border-teal-100 bg-teal-50"}`}>
      <div className="flex items-start gap-4">
        <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-xl ${highRisk ? "bg-amber-100 text-amber-800" : "bg-teal-100 text-teal-800"}`}>{highRisk ? <AlertTriangle aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />}</span>
        <div><p className="text-sm font-semibold text-slate-600">Model estimate</p><h2 className="mt-1 text-2xl font-bold text-navy-950">{highRisk ? "Higher churn risk" : "Lower churn risk"}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">This result is an estimate from the deployed model, not a certain outcome.</p></div>
      </div>
    </div>
    <div className="space-y-6 p-6 sm:p-8">
      <div>
        <div className="flex items-end justify-between gap-3"><span className="font-semibold text-navy-900">Estimated churn probability</span><span className="text-xl font-bold text-navy-950">{probability == null ? "Probability unavailable" : `${(probability * 100).toFixed(1)}%`}</span></div>
        {probability != null && <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-label="Estimated churn probability" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(probability * 100)}><div className={`h-full rounded-full ${highRisk ? "bg-amber-600" : "bg-teal-700"}`} style={{ width: `${probability * 100}%` }} /></div>}
      </div>
      <MetadataDetails entries={[["Model", result.model_name], ["Model version", result.model_version], ["Model version ID", result.model_version_id], ["Deployment ID", result.deployment_id], ["MLflow run", result.mlflow_run_id], ["Prediction timestamp", result.timestamp]]} />
      <div className="flex flex-col gap-3 sm:flex-row"><button className="button-primary" onClick={onAgain}><RotateCcw className="h-4 w-4" /> Assess another customer</button><button className="button-secondary" onClick={onEdit}><SlidersHorizontal className="h-4 w-4" /> Edit inputs</button></div>
    </div>
  </section>;
}
