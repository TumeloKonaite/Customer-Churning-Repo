import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { Controller, useForm } from "react-hook-form";
import { Send } from "lucide-react";
import { ApiErrorAlert, FieldError, FormSection, LoadingButton, PageHeader } from "../components/ui";
import { PredictionResult } from "../components/PredictionResult";
import { asApiError, predictCustomer, type ApiError } from "../lib/api";
import { defaultPredictionValues, predictionFormSchema, toPredictionRequest, type PredictionFormValues } from "../lib/prediction";
import type { PredictionResponse } from "../types/api";

type NumericName = "CreditScore" | "Age" | "Tenure" | "Balance" | "NumOfProducts" | "EstimatedSalary";

export function PredictPage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const { register, control, handleSubmit, reset, formState: { errors, isSubmitting } } = useForm<PredictionFormValues>({ resolver: zodResolver(predictionFormSchema), defaultValues: defaultPredictionValues });

  const submit = handleSubmit(async (values) => {
    setApiError(null);
    try { setResult(await predictCustomer(toPredictionRequest(values))); }
    catch (error) { setApiError(asApiError(error)); }
  });

  const numeric = (name: NumericName, label: string, hint: string, options?: { step?: string; prefix?: string }) => <div>
    <label className="label" htmlFor={name}>{label}</label>
    <div className="relative">{options?.prefix && <span className="pointer-events-none absolute inset-y-0 left-3.5 flex items-center text-sm text-slate-500">{options.prefix}</span>}<input id={name} type="number" step={options?.step ?? "1"} className={`field ${options?.prefix ? "pl-9" : ""} ${errors[name] ? "field-error" : ""}`} aria-invalid={Boolean(errors[name])} aria-describedby={`${name}-hint ${name}-error`} {...register(name, { valueAsNumber: true })} /></div>
    <p id={`${name}-hint`} className="mt-1.5 text-xs text-slate-500">{hint}</p><FieldError message={errors[name]?.message} />
  </div>;

  const select = (name: "Geography" | "Gender", label: string, choices: string[]) => <div><label className="label" htmlFor={name}>{label}</label><select id={name} className={`field ${errors[name] ? "field-error" : ""}`} aria-invalid={Boolean(errors[name])} {...register(name)}>{choices.map((choice) => <option key={choice}>{choice}</option>)}</select><FieldError message={errors[name]?.message} /></div>;

  const binary = (name: "HasCrCard" | "IsActiveMember", legend: string) => <div><span className="label">{legend}</span><Controller name={name} control={control} render={({ field }) => <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label={legend}>{[["yes", "Yes"], ["no", "No"]].map(([value, label]) => <label key={value} className={`cursor-pointer rounded-xl border px-4 py-2.5 text-center text-sm font-semibold transition focus-within:ring-2 focus-within:ring-teal-600 focus-within:ring-offset-2 ${field.value === value ? "border-teal-700 bg-teal-50 text-teal-800" : "border-slate-300 bg-white text-slate-700 hover:bg-slate-50"}`}><input type="radio" className="sr-only" name={field.name} value={value} checked={field.value === value} onChange={() => field.onChange(value)} />{label}</label>)}</div>} /></div>;

  const assessAnother = () => { reset(defaultPredictionValues); setResult(null); setApiError(null); window.scrollTo({ top: 0, behavior: "smooth" }); };

  return <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
    <PageHeader eyebrow="Single prediction" title="Assess one customer" description="Enter the customer and account information used by the deployed model. Required fields are validated before anything is sent." />
    {result ? <PredictionResult result={result} onAgain={assessAnother} onEdit={() => setResult(null)} /> : <form onSubmit={submit} noValidate className="space-y-6">
      {apiError && <ApiErrorAlert error={apiError} />}
      <div className="card space-y-8 p-6 sm:p-8">
        <FormSection title="Customer profile" description="Basic details used by the model.">{numeric("CreditScore", "Credit score", "Whole number from 300 to 850.")}{select("Geography", "Country", ["France", "Germany", "Spain"])}{select("Gender", "Gender", ["Female", "Male"])}{numeric("Age", "Age", "Whole number from 18 to 100.")}</FormSection>
        <hr />
        <FormSection title="Account relationship" description="How the customer currently engages with the bank.">{numeric("Tenure", "Tenure", "Years with the bank, from 0 to 10.")}{numeric("NumOfProducts", "Number of products", "Whole number from 1 to 4.")}{binary("HasCrCard", "Has a credit card?")}{binary("IsActiveMember", "Is an active member?")}</FormSection>
        <hr />
        <FormSection title="Financial information" description="Current balance and estimated annual salary.">{numeric("Balance", "Account balance", "Enter zero or a positive amount.", { step: "0.01", prefix: "$" })}{numeric("EstimatedSalary", "Estimated salary", "Enter zero or a positive annual amount.", { step: "0.01", prefix: "$" })}</FormSection>
      </div>
      <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end"><button type="button" className="button-secondary" onClick={() => { reset(defaultPredictionValues); setApiError(null); }}>Reset form</button><LoadingButton loading={isSubmitting} type="submit"><Send className="h-4 w-4" /> Run assessment</LoadingButton></div>
    </form>}
  </div>;
}
