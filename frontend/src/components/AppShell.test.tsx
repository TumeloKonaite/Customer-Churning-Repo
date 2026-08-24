import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";
import { HealthProvider } from "../hooks/useHealth";

describe("primary mobile layout", () => {
  it("provides a keyboard-operable mobile navigation", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ status: "healthy", model_loaded: true, metadata: {}, timestamp: "now" }), { status: 200, headers: { "Content-Type": "application/json" } }));
    render(<MemoryRouter><HealthProvider><AppShell><p>Page content</p></AppShell></HealthProvider></MemoryRouter>);
    const menu = screen.getByRole("button", { name: "Open navigation" });
    expect(menu).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(menu);
    expect(screen.getByRole("navigation", { name: "Mobile navigation" })).toBeInTheDocument();
    expect(menu).toHaveAttribute("aria-expanded", "true");
  });
});
