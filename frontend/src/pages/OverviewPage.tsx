import { ArrowRight, FileSpreadsheet, LockKeyhole, RefreshCw, ScanSearch, UserRoundCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { useHealth } from "../hooks/useHealth";

function display(value: unknown): string {
  return value === null || value === undefined || value === "" ? "Not reported" : String(value);
}

export function OverviewPage() {
  const { data, state, error, refresh } = useHealth();
  const environment = data?.metadata?.environment ?? data?.metadata?.app_env;
  const fields: Array<[string, unknown]> = [
    ["Model", data?.model_name],
    ["Version", data?.model_version],
    ["Environment", environment],
    ["Feature schema", data?.feature_schema_version],
    ["Deployment", data?.deployment_id],
    ["Integrity", data?.integrity_status],
  ];

  return <>
    <section className="border-b border-slate-200 bg-white">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-14 sm:px-6 sm:py-20 lg:grid-cols-[1.15fr_.85fr] lg:items-center lg:px-8">
        <div>
          <p className="eyebrow">Customer decision support</p>
          <h1 className="mt-3 max-w-3xl text-4xl font-bold tracking-[-0.035em] text-navy-950 sm:text-5xl lg:text-6xl">Understand churn risk earlier</h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">Use the deployed machine-learning model to assess individual customers or process a CSV batch.</p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <Link to="/predict" className="button-primary">Assess a customer <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
            <Link to="/batch" className="button-secondary"><FileSpreadsheet className="h-4 w-4" aria-hidden="true" /> Upload a batch</Link>
          </div>
          <p className="mt-6 flex max-w-2xl items-start gap-2 text-sm leading-6 text-slate-600"><ScanSearch className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" aria-hidden="true" /> Predictions are estimates and decision-support signals. They should be considered alongside appropriate human judgment.</p>
        </div>
        <div className="card overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-200 px-6 py-5">
            <div><p className="text-sm font-semibold text-navy-900">Live prediction service</p><p className="mt-0.5 text-xs text-slate-500">Reported by the configured API</p></div>
            <span className={`rounded-full px-3 py-1 text-xs font-bold capitalize ${state === "healthy" ? "bg-teal-50 text-teal-800" : state === "degraded" ? "bg-amber-50 text-amber-800" : state === "unavailable" ? "bg-red-50 text-red-800" : "bg-slate-100 text-slate-700"}`}>{state}</span>
          </div>
          {error ? <div className="p-6"><p className="text-sm font-semibold text-red-800">Health information is temporarily unavailable.</p><p className="mt-2 text-sm leading-6 text-slate-600">Prediction pages remain available. You can retry the status check independently.</p><button className="button-secondary mt-4" onClick={refresh}><RefreshCw className="h-4 w-4" /> Retry status</button></div> :
          <dl className="grid grid-cols-2 divide-x divide-y divide-slate-100">{fields.map(([label, value]) => <div className="min-w-0 p-5" key={label}><dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt><dd className="mt-1.5 truncate text-sm font-semibold text-navy-900" title={display(value)}>{display(value)}</dd></div>)}</dl>}
        </div>
      </div>
    </section>
    <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
      <div className="mb-8"><p className="eyebrow">A focused workflow</p><h2 className="mt-2 text-2xl font-bold text-navy-950 sm:text-3xl">How it works</h2></div>
      <ol className="grid gap-5 md:grid-cols-3">
        {[["01", "Enter customer data", "Provide the ten model inputs for one customer, or select a compatible CSV."], ["02", "Run the assessment", "The data is sent directly to the configured prediction API for processing."], ["03", "Review the estimate", "See the risk label, available probability, and exact model deployment metadata."]].map(([number, title, copy]) => <li className="card p-6" key={number}><span className="text-sm font-bold text-teal-700">{number}</span><h3 className="mt-4 text-lg font-bold text-navy-900">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-600">{copy}</p></li>)}
      </ol>
      <div className="mt-8 flex items-start gap-4 rounded-2xl border border-teal-100 bg-teal-50 p-5"><LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-teal-800" aria-hidden="true" /><div><h2 className="font-semibold text-navy-900">Privacy by design</h2><p className="mt-1 text-sm leading-6 text-slate-600">Form and CSV data is sent only to the configured prediction API. This interface does not persist customer inputs or send them to analytics.</p></div></div>
    </section>
  </>;
}
