import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getHealth } from "../lib/api";
import type { HealthResponse } from "../types/api";

export type ServiceState = "checking" | "healthy" | "degraded" | "unavailable";

interface HealthState {
  data: HealthResponse | null;
  state: ServiceState;
  error: Error | null;
  refresh: () => void;
}

const HealthContext = createContext<HealthState | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [checking, setChecking] = useState(true);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => {
    setChecking(true);
    setNonce((value) => value + 1);
  }, []);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const health = await getHealth();
        if (active) {
          setData(health);
          setError(null);
        }
      } catch (firstError) {
        try {
          await new Promise((resolve) => window.setTimeout(resolve, 500));
          const health = await getHealth();
          if (active) {
            setData(health);
            setError(null);
          }
        } catch {
          if (active) {
            setData(null);
            setError(firstError instanceof Error ? firstError : new Error("Health check failed"));
          }
        }
      } finally {
        if (active) setChecking(false);
      }
    };
    void load();
    const interval = window.setInterval(load, 60_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [nonce]);

  const value = useMemo<HealthState>(() => {
    let state: ServiceState = "checking";
    if (!checking && error) state = "unavailable";
    else if (!checking && data) state = data.status === "healthy" && data.model_loaded ? "healthy" : "degraded";
    return { data, error, state, refresh };
  }, [checking, data, error, refresh]);

  return <HealthContext.Provider value={value}>{children}</HealthContext.Provider>;
}

export function useHealth(): HealthState {
  const context = useContext(HealthContext);
  if (!context) throw new Error("useHealth must be used inside HealthProvider");
  return context;
}
