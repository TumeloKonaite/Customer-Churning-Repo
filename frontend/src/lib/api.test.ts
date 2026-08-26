import { afterEach, describe, expect, it, vi } from "vitest";
import { API_BASE_URL, predictBatch } from "./api";

describe("batch API request", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the JSON contract to the canonical batch endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({}),
    });
    vi.stubGlobal("fetch", fetchMock);
    const payload = {
      records: [{
        CreditScore: 650,
        Geography: "France" as const,
        Gender: "Female" as const,
        Age: 38,
        Tenure: 5,
        Balance: 75000,
        NumOfProducts: 2,
        HasCrCard: 1 as const,
        IsActiveMember: 1 as const,
        EstimatedSalary: 60000,
      }],
      options: { mode: "partial" as const },
    };

    await predictBatch(payload);

    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/predict/batch`,
      expect.objectContaining({
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  });
});
