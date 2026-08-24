import type { ReactNode } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";
import { ApiError } from "../lib/api";

export function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="mb-8 max-w-3xl"><p className="eyebrow">{eyebrow}</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-navy-950 sm:text-4xl">{title}</h1><p className="mt-3 text-base leading-7 text-slate-600 sm:text-lg">{description}</p></div>;
}

export function FormSection({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <fieldset className="border-0 p-0"><legend className="text-lg font-bold text-navy-900">{title}</legend><p className="mt-1 text-sm text-slate-600">{description}</p><div className="mt-5 grid gap-5 sm:grid-cols-2">{children}</div></fieldset>;
}

export function FieldError({ message }: { message?: string }) {
  return message ? <p className="mt-1.5 text-sm text-red-700" role="alert">{message}</p> : null;
}

export function LoadingButton({ loading, children, className = "button-primary", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { loading: boolean }) {
  return <button {...props} disabled={loading || props.disabled} className={className}>{loading && <LoaderCircle className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden="true" />}<span>{loading ? "Processing…" : children}</span></button>;
}

export function ApiErrorAlert({ error }: { error: ApiError }) {
  return <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-950" role="alert" aria-live="assertive"><div className="flex gap-3"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-700" aria-hidden="true" /><div><h2 className="font-semibold">We couldn’t complete that request</h2><p className="mt-1 text-sm leading-6">{error.message}</p>{error.details?.length ? <details className="mt-2 text-sm"><summary className="cursor-pointer font-medium">Technical details</summary><ul className="mt-2 list-disc space-y-1 pl-5">{error.details.map((detail, index) => <li key={`${detail}-${index}`}>{detail}</li>)}</ul></details> : null}</div></div></div>;
}

export function MetadataDetails({ entries }: { entries: Array<[string, unknown]> }) {
  const visible = entries.filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!visible.length) return null;
  return <details className="rounded-xl border border-slate-200 bg-slate-50 p-4"><summary className="cursor-pointer text-sm font-semibold text-navy-900">Model and deployment details</summary><dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">{visible.map(([label, value]) => <div key={label}><dt className="text-slate-500">{label}</dt><dd className="mt-0.5 break-all font-medium text-navy-900">{String(value)}</dd></div>)}</dl></details>;
}
