import { useState, type ReactNode } from "react";
import { Activity, Menu, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { ServiceStatus } from "./ServiceStatus";

const navigation = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Single prediction" },
  { to: "/batch", label: "Batch prediction" },
];

function NavItems({ onSelect }: { onSelect?: () => void }) {
  return <>{navigation.map((item) => (
    <NavLink key={item.to} to={item.to} end={item.end} onClick={onSelect} className={({ isActive }) => `rounded-lg px-3 py-2 text-sm font-medium transition ${isActive ? "bg-white/10 text-white" : "text-slate-300 hover:bg-white/5 hover:text-white"}`}>
      {item.label}
    </NavLink>
  ))}</>;
}

export function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div className="flex min-h-screen flex-col">
      <header className="bg-navy-950 text-white">
        <div className="mx-auto flex h-[72px] max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <NavLink to="/" className="flex items-center gap-3 rounded-lg" aria-label="Churn Insight home">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-teal-600"><Activity className="h-5 w-5" aria-hidden="true" /></span>
            <span><span className="block text-base font-bold leading-tight">Churn Insight</span><span className="hidden text-[11px] text-slate-400 sm:block">Prediction workspace</span></span>
          </NavLink>
          <nav className="hidden items-center gap-1 md:flex" aria-label="Primary navigation"><NavItems /></nav>
          <div className="flex items-center gap-2">
            <ServiceStatus />
            <button type="button" className="grid h-10 w-10 place-items-center rounded-lg text-slate-200 hover:bg-white/10 md:hidden" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-controls="mobile-navigation" aria-label={menuOpen ? "Close navigation" : "Open navigation"}>
              {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
        {menuOpen && <nav id="mobile-navigation" className="flex flex-col gap-1 border-t border-white/10 px-4 py-3 md:hidden" aria-label="Mobile navigation"><NavItems onSelect={() => setMenuOpen(false)} /></nav>}
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-6 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <span>Churn Insight</span><span>Decision-support estimates — not guaranteed outcomes</span>
        </div>
      </footer>
    </div>
  );
}
