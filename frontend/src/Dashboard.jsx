import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ZAxis, ResponsiveContainer, ReferenceLine
} from "recharts";

const ATTACKS = ["FORA", "FSHA", "Inverse Network"];
const DEFENSES = ["None", "NoPeekNN", "DP-Gaussian", "DP-Laplace", "DP-Laplace (mod)", "AFO"];
const ARCHITECTURES = ["Vanilla SL", "U-Shaped SL", "SplitFed"];
const CUT_LAYERS = [1, 2, 3];

const SEED_RUNS = [
  {
    id: "seed-1", attack: "Inverse Network", defense: "None",
    architecture: "Vanilla SL", cut_layer: 1,
    ssim: 0.706, psnr: 18.4, dcor: 0.733, accuracy: 78.2,
    note: "Baseline — no defense", ts: Date.now() - 86400000 * 4
  },
  {
    id: "seed-2", attack: "Inverse Network", defense: "NoPeekNN",
    architecture: "Vanilla SL", cut_layer: 1,
    ssim: 0.695, psnr: 17.8, dcor: 0.362, accuracy: 76.1,
    note: "dCor halved, SSIM barely moved", ts: Date.now() - 86400000 * 3
  },
  {
    id: "seed-3", attack: "Inverse Network", defense: "DP-Gaussian",
    architecture: "Vanilla SL", cut_layer: 1,
    ssim: 0.431, psnr: 12.1, dcor: 0.891, accuracy: 38.7,
    note: "Model collapse — dCor paradox (unreliable)", ts: Date.now() - 86400000 * 2
  },
  {
    id: "seed-4", attack: "FSHA", defense: "None",
    architecture: "Vanilla SL", cut_layer: 1,
    ssim: 0.891, psnr: 24.3, dcor: 0.811, accuracy: 78.2,
    note: "Active hijack — near-perfect reconstruction", ts: Date.now() - 86400000
  },
];

const ATTACK_COLORS = { "FORA": "#378ADD", "FSHA": "#D85A30", "Inverse Network": "#1D9E75" };
const DEFENSE_COLORS = { "None": "#888780", "NoPeekNN": "#7F77DD", "DP-Gaussian": "#D4537E", "DP-Laplace": "#BA7517", "DP-Laplace (mod)": "#EF9F27", "AFO": "#1D9E75" };

function avg(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

const mono = { fontFamily: "var(--font-mono)" };
const muted = { color: "var(--color-text-secondary)" };
const sm = { fontSize: 12 };
const xs = { fontSize: 11 };

function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "12px 14px",
      borderLeft: accent ? `3px solid ${accent}` : undefined
    }}>
      <p style={{ margin: "0 0 4px", ...sm, ...muted }}>{label}</p>
      <p style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{value}</p>
      {sub && <p style={{ margin: "3px 0 0", ...xs, ...muted, ...mono }}>{sub}</p>}
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 7px", borderRadius: 4,
      background: color + "22", color, fontSize: 11, fontWeight: 500
    }}>{text}</span>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "8px 12px", fontSize: 12
    }}>
      <p style={{ margin: "0 0 4px", fontWeight: 500 }}>{label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ margin: "2px 0", color: p.color }}>
          {p.dataKey}: {typeof p.value === "number" ? p.value.toFixed(3) : p.value}
        </p>
      ))}
    </div>
  );
};

export default function App() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("dashboard");
  const [filterAttack, setFilterAttack] = useState("All");
  const [form, setForm] = useState({
    attack: "FORA", defense: "None", architecture: "Vanilla SL",
    cut_layer: 1, ssim: "", psnr: "", dcor: "", accuracy: "", note: ""
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  useEffect(() => { loadRuns(); }, []);

  async function loadRuns() {
    try {
      const result = await window.storage.get("sl-bench-runs");
      setRuns(result ? JSON.parse(result.value) : SEED_RUNS);
      if (!result) await window.storage.set("sl-bench-runs", JSON.stringify(SEED_RUNS));
    } catch { setRuns(SEED_RUNS); }
    setLoading(false);
  }

  async function saveRun() {
    if (!form.ssim || !form.psnr || !form.dcor || !form.accuracy) return;
    setSaving(true);
    const newRun = {
      ...form,
      id: `run-${Date.now()}`,
      cut_layer: Number(form.cut_layer),
      ssim: Number(form.ssim),
      psnr: Number(form.psnr),
      dcor: Number(form.dcor),
      accuracy: Number(form.accuracy),
      ts: Date.now()
    };
    const updated = [...runs, newRun];
    try {
      await window.storage.set("sl-bench-runs", JSON.stringify(updated));
      setRuns(updated);
      setForm({ attack: "FORA", defense: "None", architecture: "Vanilla SL", cut_layer: 1, ssim: "", psnr: "", dcor: "", accuracy: "", note: "" });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) { console.error(e); }
    setSaving(false);
  }

  async function deleteRun(id) {
    const updated = runs.filter(r => r.id !== id);
    await window.storage.set("sl-bench-runs", JSON.stringify(updated));
    setRuns(updated);
    setDeleteConfirm(null);
  }

  const filteredRuns = filterAttack === "All" ? runs : runs.filter(r => r.attack === filterAttack);

  const comboData = (() => {
    const map = {};
    filteredRuns.forEach(r => {
      const key = `${r.attack}||${r.defense}`;
      if (!map[key]) map[key] = { label: `${r.attack} / ${r.defense}`, attack: r.attack, defense: r.defense, ssims: [], psnrs: [], dcors: [], accs: [] };
      map[key].ssims.push(r.ssim);
      map[key].psnrs.push(r.psnr);
      map[key].dcors.push(r.dcor);
      map[key].accs.push(r.accuracy);
    });
    return Object.values(map).map(g => ({
      label: g.label.length > 32 ? g.label.slice(0, 30) + "…" : g.label,
      fullLabel: g.label,
      attack: g.attack, defense: g.defense,
      ssim: +avg(g.ssims).toFixed(3),
      psnr: +avg(g.psnrs).toFixed(1),
      dcor: +avg(g.dcors).toFixed(3),
      accuracy: +avg(g.accs).toFixed(1),
      n: g.ssims.length
    })).sort((a, b) => a.ssim - b.ssim);
  })();

  const scatterData = filteredRuns.map(r => ({
    x: +r.accuracy.toFixed(1),
    y: +r.ssim.toFixed(3),
    attack: r.attack,
    defense: r.defense,
    note: r.note
  }));

  const totalRuns = runs.length;
  const minSSIMRun = runs.length ? runs.reduce((b, r) => r.ssim < b.ssim ? r : b, runs[0]) : null;
  const maxDCorReduction = (() => {
    const baseline = runs.find(r => r.defense === "None");
    if (!baseline) return null;
    const best = runs.reduce((b, r) => r.defense !== "None" && r.dcor < b.dcor ? r : b, { dcor: baseline.dcor, defense: "None" });
    return best.defense !== "None" ? { defense: best.defense, delta: (baseline.dcor - best.dcor).toFixed(3) } : null;
  })();

  const TABS = ["dashboard", "log run", "all runs"];

  if (loading) return (
    <div style={{ padding: "2rem", ...muted, ...sm, ...mono }}>loading runs...</div>
  );

  return (
    <div style={{ fontFamily: "var(--font-sans)", padding: "1rem 0", maxWidth: 800 }}>
      <h2 className="sr-only">SL-BENCH: Split Learning Benchmarking Dashboard</h2>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: "1.5rem" }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 500, letterSpacing: "-0.02em" }}>SL-BENCH</h1>
        <span style={{ fontSize: 13, ...muted, ...mono }}>privacy attack & defense evaluation</span>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, marginBottom: "1.5rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "6px 16px", fontSize: 13, border: "none", background: "none",
            cursor: "pointer", borderBottom: tab === t ? "2px solid var(--color-text-primary)" : "2px solid transparent",
            color: tab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)",
            fontFamily: "var(--font-sans)", transition: "color 0.15s"
          }}>{t}</button>
        ))}
      </div>

      {/* ── DASHBOARD ── */}
      {tab === "dashboard" && (
        <div>
          {/* Summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 10, marginBottom: "1.5rem" }}>
            <MetricCard label="total runs" value={totalRuns} />
            <MetricCard
              label="best reconstruction defense"
              value={minSSIMRun?.defense ?? "—"}
              sub={minSSIMRun ? `SSIM ${minSSIMRun.ssim.toFixed(3)} · ${minSSIMRun.attack}` : ""}
              accent={minSSIMRun ? DEFENSE_COLORS[minSSIMRun.defense] : undefined}
            />
            <MetricCard
              label="max dCor reduction"
              value={maxDCorReduction ? `−${maxDCorReduction.delta}` : "—"}
              sub={maxDCorReduction?.defense ?? ""}
              accent="#1D9E75"
            />
          </div>

          {/* Attack filter */}
          <div style={{ display: "flex", gap: 6, marginBottom: "1rem", alignItems: "center" }}>
            <span style={{ fontSize: 12, ...muted }}>filter:</span>
            {["All", ...ATTACKS].map(a => (
              <button key={a} onClick={() => setFilterAttack(a)} style={{
                fontSize: 11, padding: "3px 10px", cursor: "pointer",
                borderRadius: 4, border: "0.5px solid",
                borderColor: filterAttack === a ? "var(--color-border-primary)" : "var(--color-border-tertiary)",
                background: filterAttack === a ? "var(--color-background-secondary)" : "none",
                color: filterAttack === a ? "var(--color-text-primary)" : "var(--color-text-secondary)"
              }}>{a}</button>
            ))}
          </div>

          {comboData.length === 0 ? (
            <p style={{ ...sm, ...muted }}>no runs match this filter.</p>
          ) : (
            <>
              {/* SSIM bar chart */}
              <p style={{ ...xs, ...muted, margin: "0 0 8px" }}>avg ssim by configuration — lower = better defense</p>
              <div style={{ height: Math.max(180, comboData.length * 44 + 60) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comboData} layout="vertical" margin={{ left: 160, right: 50, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(1)} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={155} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine x={0.5} stroke="rgba(128,128,128,0.4)" strokeDasharray="4 3" label={{ value: "0.5", fontSize: 10, fill: "var(--color-text-secondary)" }} />
                    <Bar dataKey="ssim" fill="#378ADD" radius={[0, 3, 3, 0]}
                      label={{ position: "right", fontSize: 10, formatter: v => v.toFixed(3), fill: "var(--color-text-secondary)" }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* dCor bar chart */}
              <p style={{ ...xs, ...muted, margin: "1.5rem 0 8px" }}>avg distance correlation — lower = less privacy leakage</p>
              <div style={{ height: Math.max(180, comboData.length * 44 + 60) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comboData} layout="vertical" margin={{ left: 160, right: 50, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(1)} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={155} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="dcor" fill="#D85A30" radius={[0, 3, 3, 0]}
                      label={{ position: "right", fontSize: 10, formatter: v => v.toFixed(3), fill: "var(--color-text-secondary)" }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Privacy-Utility scatter */}
              <p style={{ ...xs, ...muted, margin: "1.5rem 0 8px" }}>privacy-utility tradeoff — lower ssim + higher accuracy = ideal defense (bottom-right)</p>
              <div style={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ left: 20, right: 20, top: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
                    <XAxis type="number" dataKey="x" name="accuracy %" domain={[0, 100]} tick={{ fontSize: 11 }} label={{ value: "accuracy (%)", position: "insideBottom", offset: -12, fontSize: 11, fill: "var(--color-text-secondary)" }} />
                    <YAxis type="number" dataKey="y" name="ssim" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "ssim", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--color-text-secondary)" }} />
                    <ZAxis range={[50, 50]} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0]?.payload;
                      return (
                        <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)", padding: "8px 12px", fontSize: 11 }}>
                          <p style={{ margin: "0 0 2px", fontWeight: 500 }}>{d.attack} / {d.defense}</p>
                          <p style={{ margin: "1px 0", ...muted }}>accuracy: {d.x}% · ssim: {d.y}</p>
                          {d.note && <p style={{ margin: "2px 0 0", ...muted, fontStyle: "italic" }}>{d.note}</p>}
                        </div>
                      );
                    }} />
                    {ATTACKS.map(atk => (
                      <Scatter key={atk} name={atk}
                        data={scatterData.filter(d => d.attack === atk)}
                        fill={ATTACK_COLORS[atk]} opacity={0.85} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: "flex", gap: 14, marginTop: 4 }}>
                {ATTACKS.map(a => (
                  <span key={a} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, ...muted }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: ATTACK_COLORS[a], display: "inline-block" }} />
                    {a}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── LOG RUN ── */}
      {tab === "log run" && (
        <div style={{ maxWidth: 480 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Field label="attack">
              <select value={form.attack} onChange={e => setForm({ ...form, attack: e.target.value })}>
                {ATTACKS.map(a => <option key={a}>{a}</option>)}
              </select>
            </Field>
            <Field label="defense">
              <select value={form.defense} onChange={e => setForm({ ...form, defense: e.target.value })}>
                {DEFENSES.map(d => <option key={d}>{d}</option>)}
              </select>
            </Field>
            <Field label="architecture">
              <select value={form.architecture} onChange={e => setForm({ ...form, architecture: e.target.value })}>
                {ARCHITECTURES.map(a => <option key={a}>{a}</option>)}
              </select>
            </Field>
            <Field label="cut layer">
              <select value={form.cut_layer} onChange={e => setForm({ ...form, cut_layer: Number(e.target.value) })}>
                {CUT_LAYERS.map(l => <option key={l}>{l}</option>)}
              </select>
            </Field>
            <Field label="ssim (0–1)">
              <input type="number" step="0.001" min="0" max="1" placeholder="0.000"
                value={form.ssim} onChange={e => setForm({ ...form, ssim: e.target.value })} />
            </Field>
            <Field label="psnr (dB)">
              <input type="number" step="0.1" placeholder="0.0"
                value={form.psnr} onChange={e => setForm({ ...form, psnr: e.target.value })} />
            </Field>
            <Field label="distance correlation (0–1)">
              <input type="number" step="0.001" min="0" max="1" placeholder="0.000"
                value={form.dcor} onChange={e => setForm({ ...form, dcor: e.target.value })} />
            </Field>
            <Field label="test accuracy (%)">
              <input type="number" step="0.1" min="0" max="100" placeholder="0.0"
                value={form.accuracy} onChange={e => setForm({ ...form, accuracy: e.target.value })} />
            </Field>
            <Field label="notes (optional)" span>
              <input type="text" placeholder="e.g. dCor paradox under model collapse…"
                value={form.note} onChange={e => setForm({ ...form, note: e.target.value })} />
            </Field>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 20 }}>
            <button onClick={saveRun}
              disabled={saving || !form.ssim || !form.psnr || !form.dcor || !form.accuracy}
              style={{ padding: "8px 20px", fontSize: 13, cursor: "pointer" }}>
              {saving ? "saving…" : "log run ↗"}
            </button>
            {saved && <span style={{ fontSize: 12, color: "#1D9E75" }}>saved.</span>}
          </div>

          <div style={{ marginTop: 24, padding: "12px 14px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
            <p style={{ margin: "0 0 4px", ...xs, fontWeight: 500 }}>metric reference</p>
            <p style={{ margin: "2px 0", ...xs, ...muted }}>ssim · higher = better reconstruction = weaker defense</p>
            <p style={{ margin: "2px 0", ...xs, ...muted }}>psnr · higher = better reconstruction quality (dB)</p>
            <p style={{ margin: "2px 0", ...xs, ...muted }}>dcor · lower = smashed data is more statistically independent from input</p>
            <p style={{ margin: "2px 0", ...xs, ...muted }}>accuracy · higher = better model utility; target: &gt;95% of baseline</p>
          </div>
        </div>
      )}

      {/* ── ALL RUNS ── */}
      {tab === "all runs" && (
        <div>
          {runs.length === 0 ? (
            <p style={{ ...sm, ...muted }}>no runs logged yet.</p>
          ) : (
            <>
              <p style={{ ...xs, ...muted, margin: "0 0 12px" }}>{runs.length} run{runs.length !== 1 ? "s" : ""} — sorted newest first</p>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ borderBottom: "0.5px solid var(--color-border-secondary)" }}>
                      {["attack", "defense", "arch", "cut", "ssim", "psnr", "dcor", "acc %", "notes", ""].map(h => (
                        <th key={h} style={{ padding: "6px 10px", textAlign: "left", ...xs, ...muted, fontWeight: 500, whiteSpace: "nowrap" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...runs].sort((a, b) => b.ts - a.ts).map(r => (
                      <tr key={r.id} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                        <td style={{ padding: "7px 10px" }}>
                          <Badge text={r.attack} color={ATTACK_COLORS[r.attack] ?? "#888"} />
                        </td>
                        <td style={{ padding: "7px 10px" }}>
                          <Badge text={r.defense} color={DEFENSE_COLORS[r.defense] ?? "#888"} />
                        </td>
                        <td style={{ padding: "7px 10px", ...xs, ...muted }}>{r.architecture?.replace(" SL", "")}</td>
                        <td style={{ padding: "7px 10px", ...xs, ...muted, ...mono }}>L{r.cut_layer}</td>
                        <td style={{ padding: "7px 10px", ...mono, ...xs }}>{r.ssim?.toFixed(3)}</td>
                        <td style={{ padding: "7px 10px", ...mono, ...xs }}>{r.psnr?.toFixed(1)}</td>
                        <td style={{ padding: "7px 10px", ...mono, ...xs }}>{r.dcor?.toFixed(3)}</td>
                        <td style={{ padding: "7px 10px", ...mono, ...xs }}>{r.accuracy?.toFixed(1)}</td>
                        <td style={{ padding: "7px 10px", ...xs, ...muted, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.note}</td>
                        <td style={{ padding: "7px 10px" }}>
                          {deleteConfirm === r.id ? (
                            <span style={{ display: "flex", gap: 6 }}>
                              <button onClick={() => deleteRun(r.id)} style={{ fontSize: 11, cursor: "pointer", color: "var(--color-text-danger)", border: "none", background: "none", padding: 0 }}>confirm</button>
                              <button onClick={() => setDeleteConfirm(null)} style={{ fontSize: 11, cursor: "pointer", ...muted, border: "none", background: "none", padding: 0 }}>cancel</button>
                            </span>
                          ) : (
                            <button onClick={() => setDeleteConfirm(r.id)} style={{ fontSize: 11, cursor: "pointer", ...muted, border: "none", background: "none", padding: 0 }}>✕</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}