import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { HealthProvider } from "../hooks/useHealth";

export function renderPage(page: ReactElement) {
  return render(<MemoryRouter><HealthProvider>{page}</HealthProvider></MemoryRouter>);
}
