import { AlertTriangle, CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import { useHealth, type ServiceState } from "../hooks/useHealth";

const states: Record<ServiceState, { label: string; classes: string; icon: typeof CheckCircle2 }> = {
  checking: { label: "Checking", classes: "bg-slate-100 text-slate-700", icon: CircleDashed },
  healthy: { label: "Healthy", classes: "bg-teal-50 text-teal-800", icon: CheckCircle2 },
  degraded: { label: "Degraded", classes: "bg-amber-50 text-amber-800", icon: AlertTriangle },
  unavailable: { label: "Unavailable", classes: "bg-red-50 text-red-800", icon: XCircle },
};

export function ServiceStatus() {
  const { state } = useHealth();
  const config = states[state];
  const Icon = config.icon;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ${config.classes}`} role="status" aria-label={`Service status: ${config.label}`}>
      <Icon className={`h-3.5 w-3.5 ${state === "checking" ? "animate-spin motion-reduce:animate-none" : ""}`} aria-hidden="true" />
      <span className="hidden sm:inline">Service</span> {config.label}
    </span>
  );
}
