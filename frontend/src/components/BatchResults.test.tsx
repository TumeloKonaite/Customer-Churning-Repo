import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BatchResults } from "./BatchResults";
import type { BatchResponse } from "../types/api";

describe("batch results", () => {
  it("renders predictions and row errors from a partial response", () => {
    const partial: BatchResponse = {
      status: "partial",
      results: [{ index: 0, id: "customer-1", predicted_label: 1, p_churn: 0.8 }],
      errors: [{ row_index: 1, id: "customer-2", field: "Age", message: "Value is invalid" }],
      summary: { total_records: 2, valid_records: 1, invalid_records: 1, error_count: 1, mode: "partial" },
      metadata: { model_name: "churn_predictor", model_version: "7" },
      timestamp: "2026-08-24T10:00:00Z",
    };
    render(<BatchResults response={partial} />);
    expect(screen.getByRole("heading", { name: /completed with some errors/i })).toBeInTheDocument();
    expect(screen.getByText("customer-1")).toBeInTheDocument();
    expect(screen.getByText("Value is invalid")).toBeInTheDocument();
  });
});
