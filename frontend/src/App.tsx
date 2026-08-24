import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { HealthProvider } from "./hooks/useHealth";
import { BatchPage } from "./pages/BatchPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PredictPage } from "./pages/PredictPage";

function NotFoundPage() {
  return <div className="mx-auto max-w-3xl px-4 py-24 text-center"><p className="eyebrow">404</p><h1 className="mt-3 text-3xl font-bold text-navy-950">Page not found</h1><p className="mt-3 text-slate-600">The page you requested does not exist in this workspace.</p><Link className="button-primary mt-7" to="/">Return to overview</Link></div>;
}

export function App() {
  return <BrowserRouter><HealthProvider><AppShell><Routes><Route path="/" element={<OverviewPage />} /><Route path="/predict" element={<PredictPage />} /><Route path="/batch" element={<BatchPage />} /><Route path="*" element={<NotFoundPage />} /></Routes></AppShell></HealthProvider></BrowserRouter>;
}
