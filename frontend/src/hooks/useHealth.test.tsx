import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ServiceStatus } from "../components/ServiceStatus";
import { HealthProvider } from "./useHealth";

describe("health endpoint failure", () => {
  it("reports unavailable without crashing its children", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("network down"));
    render(<HealthProvider><ServiceStatus /><p>Predictions remain available</p></HealthProvider>);
    expect(screen.getByText("Predictions remain available")).toBeInTheDocument();
    expect(await screen.findByText("Unavailable", {}, { timeout: 2000 })).toBeInTheDocument();
  });
});
