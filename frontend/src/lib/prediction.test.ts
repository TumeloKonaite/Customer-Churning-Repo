import { describe, expect, it } from "vitest";
import { defaultPredictionValues, predictionFormSchema, toPredictionRequest } from "./prediction";

describe("prediction form contract", () => {
  it("validates supported ranges", () => {
    expect(predictionFormSchema.safeParse(defaultPredictionValues).success).toBe(true);
    expect(predictionFormSchema.safeParse({ ...defaultPredictionValues, CreditScore: 299 }).success).toBe(false);
    expect(predictionFormSchema.safeParse({ ...defaultPredictionValues, Age: 101 }).success).toBe(false);
    expect(predictionFormSchema.safeParse({ ...defaultPredictionValues, Balance: -1 }).success).toBe(false);
  });

  it("maps friendly yes/no values to API integers", () => {
    expect(toPredictionRequest({ ...defaultPredictionValues, HasCrCard: "yes", IsActiveMember: "no" })).toMatchObject({ HasCrCard: 1, IsActiveMember: 0 });
  });
});
