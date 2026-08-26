export interface HealthResponse {
  status: string;
  timestamp: string;
  model_loaded: boolean;
  metadata: Record<string, unknown>;
  deployment_id?: string | null;
  model_name?: string | null;
  model_version?: string | null;
  model_version_id?: string | null;
  mlflow_run_id?: string | null;
  feature_schema_version?: string | null;
  pipeline_sha256?: string | null;
  artifact_manifest_sha256?: string | null;
  integrity_status?: string | null;
}

export interface PredictionRequest {
  CreditScore: number;
  Geography: "France" | "Germany" | "Spain";
  Gender: "Female" | "Male";
  Age: number;
  Tenure: number;
  Balance: number;
  NumOfProducts: number;
  HasCrCard: 0 | 1;
  IsActiveMember: 0 | 1;
  EstimatedSalary: number;
}

export interface PredictionResponse {
  status: "success";
  predicted_label: number;
  p_churn: number | null;
  model_name: string;
  model_version?: string | null;
  deployment_id?: string | null;
  model_version_id?: string | null;
  mlflow_run_id?: string | null;
  timestamp: string;
}

export type BatchMode = "partial" | "fail_fast";

export interface BatchPredictionRecord extends PredictionRequest {
  customer_id?: string | number | null;
  row_id?: string | number | null;
  id?: string | number | null;
}

export interface BatchPredictionRequest {
  records: BatchPredictionRecord[];
  options?: { mode: BatchMode };
}

export interface BatchResultItem {
  index: number;
  id: unknown | null;
  predicted_label: number;
  p_churn: number | null;
}

export interface BatchValidationError {
  row_index: number;
  id?: unknown | null;
  message: string;
  field?: string | null;
}

export interface BatchResponse {
  status: "success" | "partial" | "failed" | "error";
  results: BatchResultItem[];
  errors: BatchValidationError[] | null;
  summary: {
    total_records: number;
    valid_records: number;
    invalid_records: number;
    error_count: number;
    mode: BatchMode;
  };
  metadata: {
    model_name: string;
    model_version?: string | null;
    deployment_id?: string | null;
    model_version_id?: string | null;
    mlflow_run_id?: string | null;
    [key: string]: unknown;
  };
  timestamp: string;
}
