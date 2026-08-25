import { Braces, Play, RotateCcw } from "lucide-react";
import { useState } from "react";

import { BatchResults } from "../components/BatchResults";
import { ApiErrorAlert, LoadingButton, PageHeader } from "../components/ui";
import { ApiError, asApiError, predictBatch } from "../lib/api";
import type { BatchPredictionRequest, BatchResponse } from "../types/api";

const SAMPLE_REQUEST: BatchPredictionRequest = {
  records: [
    {
      customer_id: "CUST_001",
      CreditScore: 650,
      Geography: "France",
      Gender: "Female",
      Age: 38,
      Tenure: 5,
      Balance: 75000,
      NumOfProducts: 2,
      HasCrCard: 1,
      IsActiveMember: 1,
      EstimatedSalary: 60000,
    },
  ],
  options: { mode: "partial" },
};

const SAMPLE_JSON = JSON.stringify(SAMPLE_REQUEST, null, 2);

function parseRequest(value: string): BatchPredictionRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new ApiError(
      "The batch request must contain valid JSON.",
      400,
      undefined,
      "validation",
    );
  }
  if (
    !parsed
    || typeof parsed !== "object"
    || !Array.isArray((parsed as { records?: unknown }).records)
  ) {
    throw new ApiError(
      "The batch request must be an object containing a records array.",
      400,
      undefined,
      "validation",
    );
  }
  return parsed as BatchPredictionRequest;
}

export function BatchPage() {
  const [requestJson, setRequestJson] = useState(SAMPLE_JSON);
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const reset = () => {
    setRequestJson(SAMPLE_JSON);
    setApiError(null);
    setResult(null);
  };

  const process = async () => {
    setLoading(true);
    setApiError(null);
    setResult(null);
    try {
      setResult(await predictBatch(parseRequest(requestJson)));
    } catch (error) {
      setApiError(asApiError(error));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
      <PageHeader
        eyebrow="Batch prediction"
        title="Assess a customer batch"
        description="Send the canonical JSON batch contract with at most 100 customer records."
      />
      <section className="card p-5 sm:p-7">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold text-navy-900">
              <Braces className="h-5 w-5 text-teal-700" aria-hidden="true" /> JSON request
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Use the same request shape documented by <code>/api/predict/batch</code>.
            </p>
          </div>
          <button type="button" className="button-secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" /> Restore sample
          </button>
        </div>
        <label htmlFor="batch-json" className="sr-only">Batch request JSON</label>
        <textarea
          id="batch-json"
          className="field mt-5 min-h-[420px] font-mono text-sm leading-6"
          spellCheck={false}
          value={requestJson}
          onChange={(event) => setRequestJson(event.target.value)}
        />
        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs leading-5 text-slate-500">
            Supported modes are <code>partial</code> and <code>fail_fast</code>.
          </p>
          <LoadingButton loading={loading} onClick={process} disabled={!requestJson.trim()}>
            <Play className="h-4 w-4" /> Run batch
          </LoadingButton>
        </div>
      </section>
      {apiError && <div className="mt-6"><ApiErrorAlert error={apiError} /></div>}
      {result && <div className="mt-8"><BatchResults response={result} /></div>}
    </div>
  );
}
