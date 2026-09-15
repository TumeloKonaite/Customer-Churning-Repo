import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { PredictPage } from "./PredictPage";

function renderPage() { return render(<MemoryRouter><PredictPage /></MemoryRouter>); }

const response = {
  status: "success",
  predicted_label: 1,
  p_churn: 0.734,
  model_name: "churn_predictor",
  model_version: "7",
  deployment_id: "deployment-1",
  timestamp: "2026-08-24T10:00:00Z",
};

describe("single prediction page", () => {
  it("submits a valid assessment and shows the estimate", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(response), { status: 200, headers: { "Content-Type": "application/json" } }));
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /run assessment/i }));
    expect(await screen.findByRole("heading", { name: "Higher churn risk" })).toBeInTheDocument();
    expect(screen.getByText("73.4%")).toBeInTheDocument();
    const payload = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(payload.HasCrCard).toBe(1);
    expect(payload.IsActiveMember).toBe(1);
  });

  it("keeps the result usable when probability is null", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ ...response, predicted_label: 0, p_churn: null }), { status: 200, headers: { "Content-Type": "application/json" } }));
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /run assessment/i }));
    expect(await screen.findByRole("heading", { name: "Lower churn risk" })).toBeInTheDocument();
    expect(screen.getByText("Probability unavailable")).toBeInTheDocument();
  });

  it("shows field validation before sending", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    renderPage();
    const score = screen.getByLabelText("Credit score");
    fireEvent.change(score, { target: { value: "200" } });
    await userEvent.click(screen.getByRole("button", { name: /run assessment/i }));
    expect(await screen.findByText(/greater than or equal to 300/i)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("presents FastAPI validation failures", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: [{ loc: ["body", "Age"], msg: "Input should be a valid integer" }] }), { status: 422, headers: { "Content-Type": "application/json" } }));
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /run assessment/i }));
    await waitFor(() => expect(screen.getByText("Some information needs your attention.")).toBeInTheDocument());
    expect(screen.getByText(/Age: Input should be a valid integer/)).toBeInTheDocument();
  });
});
