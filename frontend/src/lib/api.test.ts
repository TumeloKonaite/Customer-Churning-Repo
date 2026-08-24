import { describe, expect, it } from "vitest";
import { createBatchFormData } from "./api";

describe("batch API request", () => {
  it("constructs multipart data with the file and serialized mode", () => {
    const file = new File(["CreditScore\n650\n"], "customers.csv", { type: "text/csv" });
    const data = createBatchFormData(file, "partial");
    expect(data.get("file")).toBe(file);
    expect(data.get("options")).toBe('{"mode":"partial"}');
  });
});
