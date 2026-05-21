import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ZAxis, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────
// Change this if Flask is running on a different port or host.
const API_BASE = "http://localhost:5001/api";

// ── Constants ─────────────────────────────────────────────────────────────────
const ATTACK_COLORS  = { "FORA": "#378ADD", "FSHA": "#D85A30", "Inverse Network": "#1D9E75" };
const DEFENSE_COLORS = { "None": "#888780", "NoPeekNN": "#7F77DD", "DP-Gaussian": "#D4537E", "DP-Laplace": "#BA7517", "AFO": "#1D9E75" };
const STATUS_COLORS  = { complete: "#1D9E75", running: "#378ADD", pending: "#BA7517", failed: "#D85A30" };
const STATUS_ICONS   = { complete: "✓", running: "●", pending: "○", failed: "✕" };

// ── API helper ────────────────────────────────────────────────────────────────
async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg  = Array.isArray(body.errors)
      ? body.errors.join(", ")
      : (body.error || res.statusText);
    throw new Error(msg);
  }
  return res.json();
}

// ── Style helpers ─────────────────────────────────────────────────────────────
const mono  = { fontFamily: "var(--font-mono)" };
const muted = { color: "var(--color-text-secondary)" };
const sm    = { fontSize: 12 };
const xs    = { fontSize: 11 };

// ── Sub-components ────────────────────────────────────────────────────────────
function Badge({ text, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: (color || "#888") + "22", color: color || "#888",
      fontSize: 11, fontWeight: 500, whiteSpace: "nowrap",
    }}>{text}</span>
  );
}

function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "12px 14px",
      borderLeft: accent ? `3px solid ${accent}` : undefined,
    }}>
      <p style={{ margin: "0 0 4px", ...sm, ...muted }}>{label}</p>
      <p style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{value}</p>
      {sub && <p style={{ margin: "2px 0 0", ...xs, ...muted, ...mono }}>{sub}</p>}
    </div>
  );
}

function Field({ label, children, span }) {
  return (
    <div style={span ? { gridColumn: "span 2" } : {}}>
      <label style={{ display: "block", ...xs, ...muted, marginBottom: 4 }}>{label}</label>
      {children}
    </div>
  );
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "8px 12px", fontSize: 11,
    }}>
      <p style={{ margin: "0 0 4px", fontWeight: 500, fontSize: 12 }}>{d?.label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ margin: "2px 0", color: p.color }}>
          {p.dataKey}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
          {d?.n > 1 ? ` (n=${d.n})` : ""}
        </p>
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function App() {
  const [schema,        setSchema]        = useState(null);
  const [runs,          setRuns]          = useState([]);
  const [stats,         setStats]         = useState(null);
  const [runnerStatus,  setRunnerStatus]  = useState({ current: null, queued: 0 });
  const [tab,           setTab]           = useState("dashboard");
  const [loading,       setLoading]       = useState(true);
  const [connError,     setConnError]     = useState(null);
  const [refreshing,    setRefreshing]    = useState(false);
  const [filterAttack,  setFilterAttack]  = useState("All");
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [editingNote,   setEditingNote]   = useState(null); // { id, value }
  const [activeRun,     setActiveRun]     = useState(null);
  const [submitError,   setSubmitError]   = useState(null);
  const [form, setForm] = useState({
    attack: "Inverse Network", defense: "None",
    architecture: "Vanilla SL", cut_layer: 2, epochs: 15, note: "",
  });

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  useEffect(() => { bootstrap(); }, []);

  // ── Poll runner status every 4 s ───────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      api("/status").then(setRunnerStatus).catch(() => {});
    }, 4000);
    return () => clearInterval(id);
  }, []);

  // ── Poll active run every 3 s ──────────────────────────────────────────────
  useEffect(() => {
    const rid = activeRun?._id;
    const st  = activeRun?.status;
    if (!rid || st === "complete" || st === "failed") return;
    const id = setInterval(async () => {
      try {
        const run = await api(`/runs/${rid}`);
        setActiveRun(run);
        if (run.status === "complete" || run.status === "failed") {
          refreshAll();
        }
      } catch {}
    }, 3000);
    return () => clearInterval(id);
  }, [activeRun?._id, activeRun?.status]);

  async function bootstrap() {
    try {
      const [schemaData, statsData, runsData, statusData] = await Promise.all([
        api("/schema"), api("/stats"), api("/runs"), api("/status"),
      ]);
      setSchema(schemaData);
      setStats(statsData);
      setRuns(runsData);
      setRunnerStatus(statusData);
      setForm(f => ({ ...f, epochs: schemaData.epoch_range.default }));
    } catch (e) {
      setConnError(e.message);
    }
    setLoading(false);
  }

  async function refreshAll() {
    setRefreshing(true);
    try {
      const [statsData, runsData, statusData] = await Promise.all([
        api("/stats"), api("/runs"), api("/status"),
      ]);
      setStats(statsData);
      setRuns(runsData);
      setRunnerStatus(statusData);
    } catch {}
    setRefreshing(false);
  }

  async function submitRun() {
    setSubmitError(null);
    try {
      const { run_id } = await api("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const run = await api(`/runs/${run_id}`);
      setActiveRun(run);
      setTab("submit");
    } catch (e) {
      setSubmitError(e.message);
    }
  }

  async function deleteRun(id) {
    try {
      await api(`/runs/${id}`, { method: "DELETE" });
      setRuns(r => r.filter(x => x._id !== id));
      setDeleteConfirm(null);
    } catch (e) { alert(e.message); }
  }

  async function bulkDelete(status) {
    if (!confirm(`Delete all ${status} runs?`)) return;
    try {
      const { deleted } = await api(`/runs?status=${status}`, { method: "DELETE" });
      await refreshAll();
      alert(`Deleted ${deleted} run(s).`);
    } catch (e) { alert(e.message); }
  }

  async function saveNote(id) {
    try {
      const updated = await api(`/runs/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note: editingNote.value }),
      });
      setRuns(r => r.map(x => x._id === id ? updated : x));
      setEditingNote(null);
    } catch (e) { alert(e.message); }
  }

  // ── Derived ────────────────────────────────────────────────────────────────
  const isRunning   = !!runnerStatus.current;
  const chartData   = stats?.by_condition ?? [];
  const attacks     = schema?.attacks ?? ["FORA", "FSHA", "Inverse Network"];
  const scatterData = runs
    .filter(r => r.status === "complete" && r.ssim != null)
    .map(r => ({ x: r.accuracy, y: r.ssim, attack: r.attack, defense: r.defense, note: r.note }));

  // ── Loading / error states ─────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{ padding: "2rem", ...sm, ...muted, ...mono }}>
        connecting to {API_BASE}…
      </div>
    );
  }

  if (connError) {
    return (
      <div style={{ padding: "2rem" }}>
        <p style={{ ...sm, color: "#D85A30", marginBottom: 6 }}>
          ✕ Could not reach backend: {connError}
        </p>
        <p style={{ ...xs, ...muted, marginBottom: 12 }}>
          Make sure Flask is running:  <code style={mono}>cd backend && python app.py</code>
        </p>
        <button onClick={bootstrap} style={{ fontSize: 12, padding: "4px 12px", cursor: "pointer" }}>
          retry
        </button>
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "var(--font-sans)", padding: "1rem 0", maxWidth: 840 }}>

      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: "1.25rem" }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 500, letterSpacing: "-0.02em" }}>
          SL-BENCH
        </h1>
        <span style={{ fontSize: 13, ...muted, ...mono, flex: 1 }}>
          privacy attack & defense evaluation
        </span>
        <button onClick={refreshAll} disabled={refreshing} style={{
          fontSize: 11, padding: "4px 10px", cursor: "pointer",
          border: "0.5px solid var(--color-border-tertiary)", borderRadius: 4,
          background: "none", ...muted,
        }}>
          {refreshing ? "refreshing…" : "↺ refresh"}
        </button>
        <a href={`${API_BASE}/runs/export.csv`} style={{
          fontSize: 11, padding: "4px 10px", textDecoration: "none",
          border: "0.5px solid var(--color-border-tertiary)", borderRadius: 4,
          color: "var(--color-text-secondary)",
        }}>
          ↓ csv
        </a>
      </div>

      {/* ── Runner status bar ── */}
      {(isRunning || runnerStatus.queued > 0) && (
        <div style={{
          marginBottom: "1rem", padding: "8px 12px",
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderLeft: "3px solid #378ADD",
          borderRadius: "var(--border-radius-md)",
          display: "flex", alignItems: "center", gap: 10, ...sm,
        }}>
          <span style={{ color: "#378ADD" }}>●</span>
          {isRunning && (
            <span>
              Running: <strong>{runnerStatus.current.params.attack}</strong> vs{" "}
              <strong>{runnerStatus.current.params.defense}</strong>
              {" "}| cut={runnerStatus.current.params.cut_layer}
              {" "}| {runnerStatus.current.params.epochs} epochs
            </span>
          )}
          {runnerStatus.queued > 0 && (
            <span style={{ ...xs, ...muted }}>+{runnerStatus.queued} queued</span>
          )}
        </div>
      )}

      {/* ── Tabs ── */}
      <div style={{
        display: "flex", marginBottom: "1.5rem",
        borderBottom: "0.5px solid var(--color-border-tertiary)",
      }}>
        {["dashboard", "submit", "all runs"].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "6px 16px", fontSize: 13, border: "none", background: "none",
            cursor: "pointer",
            borderBottom: tab === t ? "2px solid var(--color-text-primary)" : "2px solid transparent",
            color: tab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)",
            fontFamily: "var(--font-sans)",
          }}>
            {t}
            {t === "submit" && activeRun && ["pending","running"].includes(activeRun.status) && (
              <span style={{ marginLeft: 5, color: "#378ADD", fontSize: 10 }}>●</span>
            )}
          </button>
        ))}
      </div>

      {/* ════════════════ DASHBOARD ════════════════ */}
      {tab === "dashboard" && (
        <div>
          {/* Summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: "1.5rem" }}>
            <MetricCard label="total runs"        value={stats?.counts.total    ?? 0} />
            <MetricCard label="complete"          value={stats?.counts.complete ?? 0} accent="#1D9E75" />
            <MetricCard label="failed"            value={stats?.counts.failed   ?? 0} accent="#D85A30" />
            <MetricCard label="conditions tested" value={chartData.length} />
          </div>

          {chartData.length === 0 ? (
            <div style={{
              padding: "3rem 2rem", textAlign: "center",
              background: "var(--color-background-secondary)",
              borderRadius: "var(--border-radius-md)", ...sm, ...muted,
            }}>
              No complete runs yet — submit a run to populate the charts.
            </div>
          ) : (
            <>
              {/* Attack filter */}
              <div style={{ display: "flex", gap: 6, marginBottom: "1rem", alignItems: "center" }}>
                <span style={{ ...xs, ...muted }}>filter:</span>
                {["All", ...attacks].map(a => (
                  <button key={a} onClick={() => setFilterAttack(a)} style={{
                    fontSize: 11, padding: "3px 10px", cursor: "pointer", borderRadius: 4,
                    border: "0.5px solid",
                    borderColor: filterAttack === a ? "var(--color-border-primary)" : "var(--color-border-tertiary)",
                    background:  filterAttack === a ? "var(--color-background-secondary)" : "none",
                    color:       filterAttack === a ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  }}>{a}</button>
                ))}
              </div>

              {/* SSIM chart */}
              <p style={{ ...xs, ...muted, margin: "0 0 8px" }}>
                avg ssim by condition — lower = less reconstruction = stronger defense
              </p>
              <div style={{ height: Math.max(180, chartData.length * 48 + 60) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical"
                    margin={{ left: 170, right: 54, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={165} />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine x={0.5} stroke="rgba(128,128,128,0.35)" strokeDasharray="4 3"
                      label={{ value: "0.5", fontSize: 10, fill: "var(--color-text-secondary)" }} />
                    <Bar dataKey="avg_ssim" fill="#378ADD" radius={[0, 3, 3, 0]}
                      label={{ position: "right", fontSize: 10,
                        formatter: v => v.toFixed(3), fill: "var(--color-text-secondary)" }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* dCor chart */}
              <p style={{ ...xs, ...muted, margin: "1.5rem 0 8px" }}>
                avg distance correlation — lower = smashed data leaks less about the input
              </p>
              <div style={{ height: Math.max(180, chartData.length * 48 + 60) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical"
                    margin={{ left: 170, right: 54, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={165} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="avg_dcor" fill="#D85A30" radius={[0, 3, 3, 0]}
                      label={{ position: "right", fontSize: 10,
                        formatter: v => v.toFixed(3), fill: "var(--color-text-secondary)" }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Privacy-utility scatter */}
              {scatterData.length > 0 && (
                <>
                  <p style={{ ...xs, ...muted, margin: "1.5rem 0 8px" }}>
                    privacy-utility tradeoff — bottom-right = ideal (low ssim, high accuracy)
                  </p>
                  <div style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ left: 20, right: 20, top: 10, bottom: 24 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
                        <XAxis type="number" dataKey="x" domain={[0, 100]} tick={{ fontSize: 11 }}
                          label={{ value: "accuracy (%)", position: "insideBottom", offset: -14, fontSize: 11, fill: "var(--color-text-secondary)" }} />
                        <YAxis type="number" dataKey="y" domain={[0, 1]}   tick={{ fontSize: 11 }}
                          label={{ value: "ssim", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--color-text-secondary)" }} />
                        <ZAxis range={[55, 55]} />
                        <Tooltip cursor={{ strokeDasharray: "3 3" }}
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const d = payload[0]?.payload;
                            return (
                              <div style={{
                                background: "var(--color-background-primary)",
                                border: "0.5px solid var(--color-border-secondary)",
                                borderRadius: "var(--border-radius-md)",
                                padding: "8px 12px", fontSize: 11,
                              }}>
                                <p style={{ margin: "0 0 2px", fontWeight: 500 }}>{d.attack} / {d.defense}</p>
                                <p style={{ margin: "1px 0", ...muted }}>acc: {d.x?.toFixed(1)}% · ssim: {d.y?.toFixed(4)}</p>
                                {d.note && <p style={{ margin: "2px 0 0", ...muted, fontStyle: "italic" }}>{d.note}</p>}
                              </div>
                            );
                          }} />
                        {attacks.map(atk => (
                          <Scatter key={atk} name={atk}
                            data={scatterData.filter(d => d.attack === atk)}
                            fill={ATTACK_COLORS[atk] ?? "#888"} opacity={0.85} />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: "flex", gap: 14, marginTop: 4 }}>
                    {attacks.map(a => (
                      <span key={a} style={{ display: "flex", alignItems: "center", gap: 5, ...xs, ...muted }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, display: "inline-block",
                          background: ATTACK_COLORS[a] ?? "#888" }} />
                        {a}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* ════════════════ SUBMIT ════════════════ */}
      {tab === "submit" && (
        <div style={{ maxWidth: 500 }}>

          {/* Active run card */}
          {activeRun && (
            <div style={{
              marginBottom: "1.5rem", padding: "14px 16px",
              background: "var(--color-background-secondary)",
              borderRadius: "var(--border-radius-md)",
              borderLeft: `3px solid ${STATUS_COLORS[activeRun.status] ?? "#888"}`,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ color: STATUS_COLORS[activeRun.status], ...sm }}>
                  {STATUS_ICONS[activeRun.status]}
                </span>
                <span style={{ ...sm, fontWeight: 500 }}>
                  {activeRun.attack} vs {activeRun.defense} | cut={activeRun.cut_layer} | {activeRun.epochs} epochs
                </span>
                <Badge text={activeRun.status} color={STATUS_COLORS[activeRun.status]} />
              </div>

              {activeRun.status === "running" && (
                <p style={{ ...xs, ...muted, margin: 0 }}>
                  training in progress — polling every 3 s…
                </p>
              )}

              {activeRun.status === "pending" && (
                <p style={{ ...xs, ...muted, margin: 0 }}>
                  queued — will start when the current run finishes
                </p>
              )}

              {activeRun.status === "complete" && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginTop: 4 }}>
                  {[["ssim", activeRun.ssim, 4], ["psnr", activeRun.psnr, 2], ["dcor", activeRun.dcor, 4], ["acc %", activeRun.accuracy, 2]].map(([k, v, dp]) => (
                    <div key={k}>
                      <p style={{ margin: "0 0 2px", ...xs, ...muted }}>{k}</p>
                      <p style={{ margin: 0, ...sm, ...mono, fontWeight: 500 }}>
                        {typeof v === "number" ? v.toFixed(dp) : "—"}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {activeRun.status === "failed" && (
                <p style={{ ...xs, color: "#D85A30", margin: "4px 0 0", fontFamily: "var(--font-mono)" }}>
                  {activeRun.error}
                </p>
              )}

              {(activeRun.status === "complete" || activeRun.status === "failed") && (
                <button onClick={() => setActiveRun(null)} style={{
                  marginTop: 10, fontSize: 11, cursor: "pointer",
                  border: "0.5px solid var(--color-border-tertiary)", borderRadius: 4,
                  background: "none", color: "var(--color-text-secondary)", padding: "3px 10px",
                }}>dismiss</button>
              )}
            </div>
          )}

          {/* Form — disabled while a run is in flight */}
          {(!activeRun || ["complete", "failed"].includes(activeRun?.status)) && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="attack">
                  <select value={form.attack}
                    onChange={e => setForm({ ...form, attack: e.target.value })}>
                    {(schema?.attacks ?? []).map(a => <option key={a}>{a}</option>)}
                  </select>
                </Field>
                <Field label="defense">
                  <select value={form.defense}
                    onChange={e => setForm({ ...form, defense: e.target.value })}>
                    {(schema?.defenses ?? []).map(d => <option key={d}>{d}</option>)}
                  </select>
                </Field>
                <Field label="architecture">
                  <select value={form.architecture}
                    onChange={e => setForm({ ...form, architecture: e.target.value })}>
                    {(schema?.architectures ?? []).map(a => <option key={a}>{a}</option>)}
                  </select>
                </Field>
                <Field label="cut layer">
                  <select value={form.cut_layer}
                    onChange={e => setForm({ ...form, cut_layer: Number(e.target.value) })}>
                    {(schema?.cut_layers ?? [1, 2, 3]).map(l => <option key={l}>{l}</option>)}
                  </select>
                </Field>
                <Field label={`epochs (1–${schema?.epoch_range.max ?? 200})`}>
                  <input type="number"
                    min={schema?.epoch_range.min ?? 1}
                    max={schema?.epoch_range.max ?? 200}
                    value={form.epochs}
                    onChange={e => setForm({ ...form, epochs: Number(e.target.value) })} />
                </Field>
                <Field label="note (optional)">
                  <input type="text" placeholder="e.g. clip_norm calibrated"
                    value={form.note}
                    onChange={e => setForm({ ...form, note: e.target.value })} />
                </Field>
              </div>

              {submitError && (
                <p style={{ marginTop: 10, ...xs, color: "#D85A30" }}>{submitError}</p>
              )}

              <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
                <button onClick={submitRun} style={{ padding: "8px 22px", fontSize: 13, cursor: "pointer" }}>
                  {isRunning ? "queue run ↗" : "submit run ↗"}
                </button>
                {isRunning && (
                  <span style={{ ...xs, ...muted }}>
                    will queue behind the current run
                  </span>
                )}
              </div>

              {/* Metric reference */}
              <div style={{
                marginTop: 24, padding: "12px 14px",
                background: "var(--color-background-secondary)",
                borderRadius: "var(--border-radius-md)",
              }}>
                <p style={{ margin: "0 0 6px", ...xs, fontWeight: 500 }}>metric reference</p>
                {[
                  ["ssim", "higher = better reconstruction = weaker defense"],
                  ["psnr", "higher = better reconstruction quality (dB)"],
                  ["dcor", "higher = smashed data leaks more about the input"],
                  ["acc %", "higher = better utility; target ≥95% of no-defense baseline"],
                ].map(([k, v]) => (
                  <p key={k} style={{ margin: "2px 0", ...xs, ...muted }}>
                    <span style={mono}>{k}</span> — {v}
                  </p>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ════════════════ ALL RUNS ════════════════ */}
      {tab === "all runs" && (
        <div>
          {/* Controls row */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: "1rem" }}>
            <span style={{ ...xs, ...muted }}>
              {runs.length} run{runs.length !== 1 ? "s" : ""}
            </span>
            <div style={{ flex: 1 }} />
            {runs.some(r => r.status === "failed") && (
              <button onClick={() => bulkDelete("failed")} style={{
                fontSize: 11, cursor: "pointer", color: "#D85A30",
                border: "0.5px solid #D85A3055", borderRadius: 4,
                background: "none", padding: "3px 8px",
              }}>
                clear {runs.filter(r => r.status === "failed").length} failed
              </button>
            )}
          </div>

          {runs.length === 0 ? (
            <p style={{ ...sm, ...muted }}>no runs yet.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ borderBottom: "0.5px solid var(--color-border-secondary)" }}>
                    {["attack","defense","arch","cut","ep","ssim","psnr","dcor","acc %","status","note",""].map(h => (
                      <th key={h} style={{
                        padding: "6px 8px", textAlign: "left",
                        ...xs, ...muted, fontWeight: 500, whiteSpace: "nowrap",
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {runs.map(r => (
                    <tr key={r._id} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                      <td style={{ padding: "6px 8px" }}>
                        <Badge text={r.attack} color={ATTACK_COLORS[r.attack] ?? "#888"} />
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        <Badge text={r.defense} color={DEFENSE_COLORS[r.defense] ?? "#888"} />
                      </td>
                      <td style={{ padding: "6px 8px", ...xs, ...muted }}>
                        {r.architecture?.replace(" SL","") ?? "—"}
                      </td>
                      <td style={{ padding: "6px 8px", ...xs, ...muted, ...mono }}>L{r.cut_layer}</td>
                      <td style={{ padding: "6px 8px", ...xs, ...muted, ...mono }}>{r.epochs ?? "—"}</td>
                      <td style={{ padding: "6px 8px", ...xs, ...mono }}>{r.ssim?.toFixed(4)     ?? "—"}</td>
                      <td style={{ padding: "6px 8px", ...xs, ...mono }}>{r.psnr?.toFixed(2)     ?? "—"}</td>
                      <td style={{ padding: "6px 8px", ...xs, ...mono }}>{r.dcor?.toFixed(4)     ?? "—"}</td>
                      <td style={{ padding: "6px 8px", ...xs, ...mono }}>{r.accuracy?.toFixed(1) ?? "—"}</td>
                      <td style={{ padding: "6px 8px" }}>
                        <Badge text={r.status} color={STATUS_COLORS[r.status] ?? "#888"} />
                      </td>
                      {/* Inline note editing */}
                      <td style={{ padding: "6px 8px", maxWidth: 150 }}>
                        {editingNote?.id === r._id ? (
                          <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
                            <input value={editingNote.value}
                              onChange={e => setEditingNote({ ...editingNote, value: e.target.value })}
                              onKeyDown={e => { if (e.key === "Enter") saveNote(r._id); if (e.key === "Escape") setEditingNote(null); }}
                              style={{ fontSize: 11, width: 90 }} autoFocus />
                            <button onClick={() => saveNote(r._id)}
                              style={{ fontSize: 10, cursor: "pointer", border: "none", background: "none", color: "#1D9E75" }}>✓</button>
                            <button onClick={() => setEditingNote(null)}
                              style={{ fontSize: 10, cursor: "pointer", border: "none", background: "none", ...muted }}>✕</button>
                          </span>
                        ) : (
                          <span
                            onClick={() => r.status !== "running" && setEditingNote({ id: r._id, value: r.note || "" })}
                            title={r.note || "click to add note"}
                            style={{
                              ...xs, ...muted, display: "block", overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 140,
                              cursor: r.status !== "running" ? "pointer" : "default",
                            }}>
                            {r.note || <span style={{ opacity: 0.35 }}>add note</span>}
                          </span>
                        )}
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        {r.status !== "running" && (
                          deleteConfirm === r._id ? (
                            <span style={{ display: "flex", gap: 5 }}>
                              <button onClick={() => deleteRun(r._id)}
                                style={{ ...xs, cursor: "pointer", color: "#D85A30", border: "none", background: "none", padding: 0 }}>
                                del
                              </button>
                              <button onClick={() => setDeleteConfirm(null)}
                                style={{ ...xs, cursor: "pointer", ...muted, border: "none", background: "none", padding: 0 }}>
                                no
                              </button>
                            </span>
                          ) : (
                            <button onClick={() => setDeleteConfirm(r._id)}
                              style={{ ...xs, cursor: "pointer", ...muted, border: "none", background: "none", padding: 0 }}>
                              ✕
                            </button>
                          )
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}