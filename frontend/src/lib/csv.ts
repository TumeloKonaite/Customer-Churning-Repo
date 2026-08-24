import type { BatchResponse } from "../types/api";

export const REQUIRED_CSV_HEADERS = [
  "CreditScore",
  "Geography",
  "Gender",
  "Age",
  "Tenure",
  "Balance",
  "NumOfProducts",
  "HasCrCard",
  "IsActiveMember",
  "EstimatedSalary",
] as const;

export const SAMPLE_CSV = `${REQUIRED_CSV_HEADERS.join(",")}\n650,France,Female,38,5,75000,2,1,1,60000\n`;

export async function validateCsvFile(file: File): Promise<string | null> {
  if (!file.name.toLowerCase().endsWith(".csv")) return "Choose a file with a .csv extension.";
  if (file.size === 0) return "The selected file is empty.";
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return "The selected file is empty.";
  const headers = lines[0].replace(/^\uFEFF/, "").split(",").map((value) => value.trim());
  const missing = REQUIRED_CSV_HEADERS.filter((header) => !headers.includes(header));
  if (missing.length) return `Missing required columns: ${missing.join(", ")}.`;
  if (lines.length - 1 > 100) return "This file contains more than the maximum 100 records.";
  if (lines.length === 1) return "The CSV has headers but no customer records.";
  return null;
}

function escapeCsv(value: unknown): string {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function downloadText(filename: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function resultsToCsv(response: BatchResponse): string {
  const rows = ["row_index,customer_id,predicted_label,risk,p_churn,error_field,error_message"];
  response.results.forEach((result) => {
    rows.push(
      [result.index, result.id, result.predicted_label, result.predicted_label === 1 ? "Higher" : "Lower", result.p_churn, "", ""]
        .map(escapeCsv)
        .join(","),
    );
  });
  response.errors?.forEach((error) => {
    rows.push([error.row_index, error.id, "", "", "", error.field, error.message].map(escapeCsv).join(","));
  });
  return `${rows.join("\n")}\n`;
}
