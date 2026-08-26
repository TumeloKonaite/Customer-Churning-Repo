import type {
  BatchPredictionRequest,
  BatchResponse,
  HealthResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/api";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = (configuredBaseUrl || "http://localhost:5001").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 30_000;
const BATCH_TIMEOUT_MS = 120_000;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly details?: string[],
    public readonly kind: "validation" | "timeout" | "network" | "server" = "server",
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function readableDetail(detail: unknown): string[] {
  if (!Array.isArray(detail)) return [];
  return detail.map((item) => {
    if (typeof item === "string") return item;
    if (item && typeof item === "object") {
      const value = item as { loc?: unknown[]; msg?: string };
      const field = value.loc?.filter((part) => part !== "body").join(" → ");
      return field ? `${field}: ${value.msg ?? "Invalid value"}` : value.msg ?? "Invalid value";
    }
    return String(item);
  });
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  const contentType = response.headers.get("content-type") ?? "";
  let message = `The prediction service returned an error (${response.status}).`;
  let details: string[] = [];

  if (contentType.includes("json")) {
    const body = (await response.json().catch(() => null)) as
      | { message?: string; detail?: unknown; errors?: unknown }
      | null;
    if (body?.message) message = body.message;
    details = readableDetail(body?.detail ?? body?.errors);
    if (!body?.message && details.length) message = "Some information needs your attention.";
  } else {
    const text = (await response.text().catch(() => "")).trim();
    if (text) details = [text.slice(0, 500)];
  }

  return new ApiError(
    message,
    response.status,
    details,
    response.status === 400 || response.status === 413 || response.status === 415 || response.status === 422
      ? "validation"
      : "server",
  );
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
    if (!response.ok) throw await errorFromResponse(response);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("The request took too long. Please try again.", undefined, undefined, "timeout");
    }
    throw new ApiError(
      navigator.onLine
        ? "The service could not be reached. Check that this site is allowed by the API’s CORS configuration."
        : "You appear to be offline. Reconnect and try again.",
      undefined,
      error instanceof Error ? [error.message] : undefined,
      "network",
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { headers: { Accept: "application/json" } }, 10_000);
}

export async function predictCustomer(payload: PredictionRequest): Promise<PredictionResponse> {
  return request<PredictionResponse>("/api/predict", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function predictBatch(payload: BatchPredictionRequest): Promise<BatchResponse> {
  return request<BatchResponse>(
    "/api/predict/batch",
    {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    BATCH_TIMEOUT_MS,
  );
}

export function asApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError("Something unexpected happened. Please try again.");
}
