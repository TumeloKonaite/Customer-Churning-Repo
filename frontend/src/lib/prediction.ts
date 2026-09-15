import { z } from "zod";
import type { PredictionRequest } from "../types/api";

export const predictionFormSchema = z.object({
  CreditScore: z.number({ invalid_type_error: "Enter a credit score" }).int().min(300).max(850),
  Geography: z.enum(["France", "Germany", "Spain"], { required_error: "Choose a country" }),
  Gender: z.enum(["Female", "Male"], { required_error: "Choose a gender" }),
  Age: z.number({ invalid_type_error: "Enter an age" }).int().min(18).max(100),
  Tenure: z.number({ invalid_type_error: "Enter tenure" }).int().min(0).max(10),
  Balance: z.number({ invalid_type_error: "Enter a balance" }).min(0),
  NumOfProducts: z.number({ invalid_type_error: "Enter the number of products" }).int().min(1).max(4),
  HasCrCard: z.enum(["yes", "no"]),
  IsActiveMember: z.enum(["yes", "no"]),
  EstimatedSalary: z.number({ invalid_type_error: "Enter an estimated salary" }).min(0),
});

export type PredictionFormValues = z.infer<typeof predictionFormSchema>;

export const defaultPredictionValues: PredictionFormValues = {
  CreditScore: 650,
  Geography: "France",
  Gender: "Female",
  Age: 38,
  Tenure: 5,
  Balance: 75000,
  NumOfProducts: 2,
  HasCrCard: "yes",
  IsActiveMember: "yes",
  EstimatedSalary: 60000,
};

export function toPredictionRequest(values: PredictionFormValues): PredictionRequest {
  return {
    ...values,
    HasCrCard: values.HasCrCard === "yes" ? 1 : 0,
    IsActiveMember: values.IsActiveMember === "yes" ? 1 : 0,
  };
}
