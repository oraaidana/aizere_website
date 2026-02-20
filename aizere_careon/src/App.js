import { useState, useCallback, useRef, useEffect } from "react";

// ─── API CONFIG ───────────────────────────────────────────────────────────────
const API = "  https://6f9a-5-34-127-109.ngrok-free.app";
async function apiPredict(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/predict`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

async function apiHealth() {
  const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
  return res.json();
}

// ─── RISK ─────────────────────────────────────────────────────────────
// Thresholds match backend: get_risk_meta()
// HIGH >= 0.25 | MODERATE 0.09-0.24 | LOW < 0.09
function riskMeta(prob) {
  const p = prob ?? 0;
  if (p >= 0.25) return { color: "#ff4757", label: "HIGH RISK",  band: "high" }; 
  if (p >= 0.09) return { color: "#ffa502", label: "MODERATE",   band: "mod"  }; 
  return               { color: "#2ed573", label: "LOW RISK",   band: "low"  };
}
// ─── RECOMMENDATIONS ───────────────────────────────────────────────────
function getRecommendations(probability, isComparison, delta) {
  const pct = (probability ?? 0) * 100;

  const urgent = [
    { priority:"URGENT",
      title:"Immediate Neurological Consultation",
      body:"Refer patient to a neurologist within 24 hours. Do not delay evaluation pending further workup." },
    { priority:"URGENT",
      title:"Emergency Imaging Protocol",
      body:"Order CT angiography and/or diffusion-weighted MRI immediately. Ischemic tissue may be salvageable within 4.5-hour window." },
    { priority:"URGENT",
      title:"Anticoagulation & Thrombolysis Review",
      body:"Assess candidacy for IV tPA or mechanical thrombectomy. Review current anticoagulant regimen and contraindications." },
    { priority:"HIGH",
      title:"Continuous Hemodynamic Monitoring",
      body:"Initiate continuous BP, SpO₂, and ECG monitoring. Target SBP <180/105 mmHg. Avoid hypotension." },
  ];

  const moderate = [
    { priority:"MODERATE",
      title:"Outpatient Neurology Referral",
      body:"Schedule neurology appointment within 1–2 weeks. TIA evaluation protocol may be indicated." },
    { priority:"MODERATE",
      title:"Comprehensive Metabolic Panel",
      body:"Order CBC, lipid panel, HbA1c, coagulation screen (PT/INR/aPTT), and hsCRP. Screen for atrial fibrillation (Holter)." },
    { priority:"MODERATE",
      title:"Medication Optimisation",
      body:"Review antihypertensive, antiplatelet (aspirin/clopidogrel), and statin therapy. Optimise to current ESC/AHA guidelines." },
    { priority:"LOW",
      title:"Structured Lifestyle Intervention",
      body:"Initiate Mediterranean dietary plan, aerobic exercise ≥150 min/week, smoking cessation counselling, BMI optimisation." },
  ];

  const low = [
    { priority:"LOW",
      title:"Routine Annual MRI Surveillance",
      body:"Schedule follow-up neuroimaging in 12 months. Continue current preventive cardiovascular care." },
    { priority:"LOW",
      title:"Primary Prevention Programme",
      body:"Encourage Mediterranean diet adherence, physical activity, stress management, and adequate sleep hygiene." },
    { priority:"LOW",
      title:"10-Year ASCVD Risk Profiling",
      body:"Calculate Pooled Cohort Equations score. Consider low-dose aspirin if 10-year risk ≥10% and no contraindications." },
  ];

  const worseningRec = { icon:"⚠️", priority:"URGENT",
    title:"Significant Progression Detected",
    body:`Stroke risk increased by +${Math.abs(delta).toFixed(1)}% since last scan. Escalate MRI monitoring to monthly intervals. Urgent clinical reassessment required.` };

  const improvingRec = { icon:"✅", priority:"LOW",
    title:"Positive Treatment Response",
    body:`Risk reduced by ${Math.abs(delta).toFixed(1)}% since last scan. Maintain current therapeutic regimen. Schedule 3-month follow-up.` };

  let recs = pct >= 25 ? urgent : pct >= 9 ? moderate : low;

  if (isComparison) {
    if (delta > 5)       recs = [worseningRec, ...recs];
    else if (delta < -5) recs = [improvingRec, ...recs.slice(0, 2)];
  }

  return recs;
}
const _users = {};
function authRegister(name, email, password) {
  if (_users[email]) return { error: "Account already exists." };
  _users[email] = { name, email, password, id: Date.now().toString() };
  return { user: _users[email] };
}
function authLogin(email, password) {
  const u = _users[email];
  if (!u || u.password !== password) return { error: "Invalid credentials." };
  return { user: u };
}

const Icon = {
  Brain: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M9.5 2a2.5 2.5 0 0 1 5 0v.5A2.5 2.5 0 0 1 12 5a2.5 2.5 0 0 1-2.5-2.5V2z"/>
      <path d="M5 8.5A3.5 3.5 0 0 1 8.5 5H12M19 8.5A3.5 3.5 0 0 0 15.5 5H12M5 8.5a4 4 0 0 0 0 8M19 8.5a4 4 0 0 1 0 8M5 16.5A3.5 3.5 0 0 0 8.5 20H12M19 16.5A3.5 3.5 0 0 1 15.5 20H12M12 5v15"/>
    </svg>
  ),
  Upload: () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
    </svg>
  ),
  Scan: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="3" y="3" width="4" height="4" rx="1"/><rect x="17" y="3" width="4" height="4" rx="1"/>
      <rect x="3" y="17" width="4" height="4" rx="1"/><rect x="17" y="17" width="4" height="4" rx="1"/>
      <path d="M7 5h10M5 7v10M19 7v10M7 19h10"/>
    </svg>
  ),
  User: () => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
    </svg>
  ),
  Logout: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
    </svg>
  ),
  History: () => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>
    </svg>
  ),
  Chart: () => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  ),
  Check: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M20 6L9 17l-5-5"/>
    </svg>
  ),
  Alert: () => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  ),
  Arrow: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12h14M12 5l7 7-7 7"/>
    </svg>
  ),
};


const G = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&display=swap');
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --void:#050508;--deep:#080b12;--panel:#0d1120;--card:#111827;--hov:#151d2e;
      --b:rgba(99,179,237,.11);--bg:rgba(99,179,237,.28);
      --teal:#00d4c8;--blue:#4a9eff;--bdim:#2563eb;
      --red:#ff4757;--amb:#ffa502;--grn:#2ed573;
      --t1:#f0f4ff;--t2:#8896b3;--t3:#4a5568;
      --fd:'Syne',sans-serif;--fm:'DM Mono',monospace;
    }
    html,body,#root{height:100%;background:var(--void);color:var(--t1);font-family:var(--fd);-webkit-font-smoothing:antialiased}
    body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
      background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.025) 2px,rgba(0,0,0,.025) 4px)}
    ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-track{background:var(--deep)}
    ::-webkit-scrollbar-thumb{background:var(--bg);border-radius:2px}
    input:-webkit-autofill,input:-webkit-autofill:focus{
      -webkit-box-shadow:0 0 0 30px var(--card) inset!important;
      -webkit-text-fill-color:var(--t1)!important}
    @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
    @keyframes fadeIn{from{opacity:0}to{opacity:1}}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
    @keyframes ring{0%{transform:scale(1);opacity:.5}100%{transform:scale(1.9);opacity:0}}
    @keyframes scanln{0%{top:-2px}100%{top:100%}}
    @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
  `}</style>
);

// ─── AUTH PAGE ────────────────────────────────────────────────────────────────
function AuthPage({ onAuth }) {
  const [mode, setMode]     = useState("login");
  const [form, setForm]     = useState({ name:"", email:"", password:"" });
  const [error, setError]   = useState("");
  const [loading, setLoading] = useState(false);
  const [apiOk, setApiOk]   = useState(null);

  useEffect(() => {
    apiHealth().then(() => setApiOk(true)).catch(() => setApiOk(false));
  }, []);

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }));

  const submit = async e => {
    e.preventDefault(); setError(""); setLoading(true);
    await new Promise(r => setTimeout(r, 480));
    const res = mode === "login"
      ? authLogin(form.email, form.password)
      : authRegister(form.name, form.email, form.password);
    setLoading(false);
    if (res.error) { setError(res.error); return; }
    onAuth(res.user);
  };

  const apiColor = apiOk === true ? "#2ed573" : apiOk === false ? "#ff4757" : "#4a5568";

  return (
    <div style={{ minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center",
      position:"relative", overflow:"hidden",
      background:"radial-gradient(ellipse 80% 55% at 50% -5%,rgba(0,212,200,.07) 0%,transparent 70%)" }}>

      {/* bg grid */}
      <div style={{ position:"absolute", inset:0, zIndex:0,
        backgroundImage:"linear-gradient(rgba(99,179,237,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(99,179,237,.04) 1px,transparent 1px)",
        backgroundSize:"48px 48px" }} />
      {/* orbs */}
      <div style={{ position:"absolute", top:"8%", left:"4%", width:380, height:380, pointerEvents:"none",
        background:"radial-gradient(circle,rgba(0,212,200,.06) 0%,transparent 70%)" }} />
      <div style={{ position:"absolute", bottom:"10%", right:"4%", width:260, height:260, pointerEvents:"none",
        background:"radial-gradient(circle,rgba(74,158,255,.05) 0%,transparent 70%)" }} />

      <div style={{ position:"relative", zIndex:1, width:"100%", maxWidth:420,
        padding:"0 24px", animation:"fadeUp .45s ease" }}>

        {/* Logo */}
        <div style={{ textAlign:"center", marginBottom:34 }}>
          <div style={{ display:"inline-flex", alignItems:"center", gap:10, marginBottom:8 }}>
            <div style={{ width:40, height:40, borderRadius:11,
              background:"linear-gradient(135deg,var(--teal),var(--bdim))",
              display:"flex", alignItems:"center", justifyContent:"center",
              boxShadow:"0 0 26px rgba(0,212,200,.38)" }}>
              <Icon.Brain />
            </div>
            <span style={{ fontSize:26, fontWeight:800, letterSpacing:"-.5px",
              background:"linear-gradient(90deg,var(--teal),var(--blue))",
              WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>CareOn</span>
          </div>
          <div style={{ fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)", letterSpacing:2 }}>
            NEURAL STROKE DIAGNOSTIC SYSTEM
          </div>
          {/* API pill */}
          <div style={{ marginTop:10, display:"inline-flex", alignItems:"center", gap:5,
            padding:"3px 10px", borderRadius:20, fontSize:11, fontFamily:"var(--fm)",
            background:`${apiColor}10`, border:`1px solid ${apiColor}30`, color:apiColor }}>
            <div style={{ width:5, height:5, borderRadius:"50%", background:apiColor,
              animation: apiOk ? "blink 2.5s ease infinite" : "none" }} />
            {apiOk === null ? "Connecting…" : apiOk ? `Connected · ${API}` : `Offline · ${API}`}
          </div>
        </div>

        {/* Card */}
        <div style={{ background:"var(--panel)", border:"1px solid var(--b)", borderRadius:20,
          padding:"30px 26px", boxShadow:"0 24px 80px rgba(0,0,0,.52),inset 0 1px 0 rgba(255,255,255,.03)" }}>

          {/* Tab switch */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr",
            background:"var(--deep)", borderRadius:10, padding:3, marginBottom:24 }}>
            {[["login","Sign In"],["register","Register"]].map(([m,lbl]) => (
              <button key={m} onClick={() => { setMode(m); setError(""); }} style={{
                padding:"9px 0", border:"none", cursor:"pointer", borderRadius:8,
                fontFamily:"var(--fd)", fontWeight:600, fontSize:13, transition:"all .18s",
                background: mode===m
                  ? "linear-gradient(135deg,rgba(0,212,200,.13),rgba(74,158,255,.08))"
                  : "transparent",
                color: mode===m ? "var(--teal)" : "var(--t3)",
                outline: mode===m ? "1px solid rgba(0,212,200,.18)" : "none",
              }}>{lbl}</button>
            ))}
          </div>

          <form onSubmit={submit}>
            {mode === "register" &&
              <Field label="Full Name" value={form.name} onChange={set("name")}
                placeholder="Dr. Amara Osei" type="text" />}
            <Field label="Email" value={form.email} onChange={set("email")}
              placeholder="physician@hospital.org" type="email" />
            <Field label="Password" value={form.password} onChange={set("password")}
              placeholder="••••••••" type="password" />

            {error && (
              <div style={{ display:"flex", alignItems:"center", gap:7,
                background:"rgba(255,71,87,.08)", border:"1px solid rgba(255,71,87,.2)",
                borderRadius:8, padding:"9px 13px", marginBottom:14,
                color:"var(--red)", fontSize:13, fontFamily:"var(--fm)" }}>
                <Icon.Alert /> {error}
              </div>
            )}

            <button type="submit" disabled={loading} style={{
              width:"100%", padding:"13px 0", border:"none", borderRadius:11,
              cursor: loading ? "not-allowed" : "pointer",
              fontFamily:"var(--fd)", fontWeight:700, fontSize:15,
              background: loading ? "var(--hov)" : "linear-gradient(135deg,var(--teal),var(--bdim))",
              color: loading ? "var(--t3)" : "#fff",
              boxShadow: loading ? "none" : "0 4px 22px rgba(0,212,200,.26)",
              transition:"all .2s", display:"flex", alignItems:"center", justifyContent:"center", gap:8,
            }}>
              {loading
                ? <span style={{ width:17, height:17, border:"2px solid var(--t3)",
                    borderTopColor:"var(--teal)", borderRadius:"50%", animation:"spin .7s linear infinite" }} />
                : (mode === "login" ? "Access Dashboard" : "Create Account")}
            </button>
          </form>
        </div>
        <p style={{ textAlign:"center", marginTop:16, fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)" }}>
          For authorised medical personnel only
        </p>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ marginBottom:14 }}>
      <label style={{ display:"block", marginBottom:5, fontSize:11, fontWeight:600,
        color:"var(--t2)", letterSpacing:.8, textTransform:"uppercase", fontFamily:"var(--fm)" }}>
        {label}
      </label>
      <input type={type} value={value} onChange={onChange} placeholder={placeholder}
        required onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={{ width:"100%", padding:"11px 13px", background:"var(--deep)", border:"1px solid",
          borderColor: focused ? "var(--teal)" : "var(--b)", borderRadius:9,
          color:"var(--t1)", fontFamily:"var(--fm)", fontSize:14, outline:"none",
          transition:"all .18s", boxShadow: focused ? "0 0 0 3px rgba(0,212,200,.07)" : "none" }} />
    </div>
  );
}

// ─── DASHBOARD ────────────────────────────────────────────────────────────────
function Dashboard({ user, onLogout }) {
  const [view, setView]       = useState("diagnose");
  const [history, setHistory] = useState([]);
  const addResult = r => setHistory(h => [r, ...h]);
  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column" }}>
      <Navbar user={user} onLogout={onLogout} view={view} setView={setView} />
      <main style={{ flex:1, maxWidth:1160, margin:"0 auto", width:"100%", padding:"36px 24px 60px" }}>
        {view === "diagnose" && <DiagnoseView onResult={addResult} />}
        {view === "trends"   && <TrendsView history={history} />}
        {view === "history"  && <HistoryView history={history} />}
      </main>
    </div>
  );
}

function Navbar({ user, onLogout, view, setView }) {
  const tabs = [
    { id:"diagnose", icon:<Icon.Scan />,    label:"Diagnose" },
    { id:"trends",   icon:<Icon.Chart />,   label:"Risk Trend" },
    { id:"history",  icon:<Icon.History />, label:"History" },
  ];
  return (
    <nav style={{ display:"flex", alignItems:"center", justifyContent:"space-between",
      padding:"0 22px", height:58, background:"rgba(8,11,18,.96)", backdropFilter:"blur(20px)",
      borderBottom:"1px solid var(--b)", position:"sticky", top:0, zIndex:100 }}>
      <div style={{ display:"flex", alignItems:"center", gap:9 }}>
        <div style={{ width:31, height:31, borderRadius:9,
          background:"linear-gradient(135deg,var(--teal),var(--bdim))",
          display:"flex", alignItems:"center", justifyContent:"center",
          boxShadow:"0 0 13px rgba(0,212,200,.3)" }}>
          <Icon.Brain />
        </div>
        <span style={{ fontSize:18, fontWeight:800,
          background:"linear-gradient(90deg,var(--teal),var(--blue))",
          WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>CareOn</span>
      </div>
      <div style={{ display:"flex", gap:2 }}>
        {tabs.map(({ id, icon, label }) => (
          <button key={id} onClick={() => setView(id)} style={{
            display:"flex", alignItems:"center", gap:6, padding:"7px 14px",
            border:"none", borderRadius:8, cursor:"pointer",
            fontFamily:"var(--fd)", fontWeight:600, fontSize:13, transition:"all .18s",
            background: view===id ? "rgba(0,212,200,.1)" : "transparent",
            color: view===id ? "var(--teal)" : "var(--t2)",
            outline: view===id ? "1px solid rgba(0,212,200,.17)" : "none",
          }}>
            {icon} {label}
          </button>
        ))}
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:11 }}>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontSize:13, fontWeight:600 }}>{user.name}</div>
          <div style={{ fontSize:10, color:"var(--t3)", fontFamily:"var(--fm)" }}>{user.email}</div>
        </div>
        <div style={{ width:31, height:31, borderRadius:"50%",
          background:"linear-gradient(135deg,rgba(0,212,200,.14),rgba(74,158,255,.14))",
          border:"1px solid var(--bg)", display:"flex", alignItems:"center",
          justifyContent:"center", color:"var(--teal)" }}>
          <Icon.User />
        </div>
        <button onClick={onLogout} style={{ display:"flex", alignItems:"center", gap:5,
          padding:"6px 11px", border:"1px solid var(--b)", borderRadius:8,
          background:"transparent", color:"var(--t2)", cursor:"pointer",
          fontFamily:"var(--fd)", fontSize:12, fontWeight:600 }}>
          <Icon.Logout /> Out
        </button>
      </div>
    </nav>
  );
}

// ─── DIAGNOSE VIEW ────────────────────────────────────────────────────────────
function DiagnoseView({ onResult }) {
  const [mode, setMode]         = useState("single");
  const [pastFile, setPastFile] = useState(null);
  const [currFile, setCurrFile] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState("");
  const reset = () => { setResult(null); setError(""); };

  const run = async () => {
    if (!currFile) { setError("Please upload a scan."); return; }
    if (mode === "compare" && !pastFile) { setError("Upload both scans for comparison."); return; }
    setError(""); setLoading(true); setResult(null);
    try {
      const curr = await apiPredict(currFile);
      let past = null;
      if (mode === "compare" && pastFile) past = await apiPredict(pastFile);
      const r = { curr, past, mode, ts: new Date().toLocaleString(),
                  filename: currFile.name, pastFilename: pastFile?.name };
      setResult(r);
      onResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ animation:"fadeUp .4s ease" }}>
      <div style={{ marginBottom:26 }}>
        <h1 style={{ fontSize:30, fontWeight:800, letterSpacing:"-.5px", marginBottom:6 }}>
          Stroke{" "}
          <span style={{ background:"linear-gradient(90deg,var(--teal),var(--blue))",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Diagnostic</span>
        </h1>
        <p style={{ color:"var(--t2)", fontSize:13, lineHeight:1.6 }}>
          Upload NIfTI MRI → 3D-CNN inference · GradCAM heatmap · Clinical recommendations
        </p>
      </div>

      {/* Mode */}
      <div style={{ display:"flex", gap:8, marginBottom:20 }}>
        {[["single","Single Scan"],["compare","Past vs Current"]].map(([id,lbl]) => (
          <button key={id} onClick={() => { setMode(id); reset(); }} style={{
            padding:"9px 18px", border:"1px solid",
            borderColor: mode===id ? "var(--teal)" : "var(--b)", borderRadius:9,
            cursor:"pointer", fontFamily:"var(--fd)", fontWeight:600, fontSize:13,
            background: mode===id ? "rgba(0,212,200,.09)" : "transparent",
            color: mode===id ? "var(--teal)" : "var(--t2)", transition:"all .18s",
          }}>{lbl}</button>
        ))}
      </div>

      {/* Drop zones */}
      <div style={{ display:"grid",
        gridTemplateColumns: mode==="compare" ? "1fr 1fr" : "1fr",
        gap:13, marginBottom:15 }}>
        {mode === "compare" && (
          <DropZone label="Past Scan" sub="Baseline MRI (.nii / .nii.gz)"
            file={pastFile} onFile={f => { setPastFile(f); reset(); }} accent="#8b5cf6" />
        )}
        <DropZone
          label={mode==="single" ? "MRI Scan" : "Current Scan"}
          sub={mode==="single" ? ".nii or .nii.gz" : "Most recent MRI"}
          file={currFile} onFile={f => { setCurrFile(f); reset(); }} accent="var(--teal)" />
      </div>

      {error && (
        <div style={{ display:"flex", alignItems:"center", gap:7,
          background:"rgba(255,71,87,.07)", border:"1px solid rgba(255,71,87,.2)",
          borderRadius:9, padding:"10px 14px", marginBottom:13,
          color:"var(--red)", fontSize:13, fontFamily:"var(--fm)" }}>
          <Icon.Alert /> {error}
        </div>
      )}

      <button onClick={run} disabled={loading || !currFile} style={{
        display:"flex", alignItems:"center", gap:8, padding:"13px 28px",
        border:"none", borderRadius:12, marginBottom:28,
        cursor: (loading||!currFile) ? "not-allowed" : "pointer",
        fontFamily:"var(--fd)", fontWeight:700, fontSize:15,
        background: (loading||!currFile) ? "var(--card)" : "linear-gradient(135deg,var(--teal),var(--bdim))",
        color: (loading||!currFile) ? "var(--t3)" : "#fff",
        boxShadow: (loading||!currFile) ? "none" : "0 6px 28px rgba(0,212,200,.3)",
        transition:"all .22s",
      }}>
        {loading
          ? <><Spinner /> Analyzing MRI…</>
          : <><Icon.Scan /> {mode==="compare" ? "Compare Scans" : "Run Diagnosis"} <Icon.Arrow /></>}
      </button>

      {loading && <ScanAnim />}
      {result && !loading && <ResultPanel result={result} />}
    </div>
  );
}

// ─── DROP ZONE ────────────────────────────────────────────────────────────────
function DropZone({ label, sub, file, onFile, accent }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();
  const handle = useCallback(f => {
    if (f && (f.name.endsWith(".nii") || f.name.endsWith(".nii.gz"))) onFile(f);
  }, [onFile]);
  return (
    <div onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
      style={{ border:"2px dashed",
        borderColor: drag ? accent : file ? "rgba(46,213,115,.4)" : "var(--b)",
        borderRadius:14, padding:"26px 20px", cursor:"pointer", textAlign:"center",
        background: drag ? `${accent}12` : file ? "rgba(46,213,115,.04)" : "var(--panel)",
        transition:"all .22s", boxShadow: drag ? `0 0 0 3px ${accent}18` : "none" }}>
      <input ref={ref} type="file" accept=".nii,.nii.gz"
        onChange={e => handle(e.target.files[0])} style={{ display:"none" }} />
      <div style={{ width:46, height:46, borderRadius:12, margin:"0 auto 11px",
        background: file ? "rgba(46,213,115,.1)" : `${accent}18`,
        display:"flex", alignItems:"center", justifyContent:"center",
        color: file ? "var(--grn)" : accent }}>
        {file ? <Icon.Check /> : <Icon.Upload />}
      </div>
      <div style={{ fontWeight:700, fontSize:14, marginBottom:4 }}>{label}</div>
      <div style={{ fontSize:12, color:"var(--t2)", fontFamily:"var(--fm)", marginBottom:8 }}>
        {file ? file.name : sub}
      </div>
      {file
        ? <span style={{ padding:"3px 9px", background:"rgba(46,213,115,.1)", borderRadius:5,
            fontSize:11, color:"var(--grn)", fontFamily:"var(--fm)" }}>
            ✓ {(file.size/1048576).toFixed(1)} MB ready
          </span>
        : <span style={{ fontSize:12, color:"var(--t3)" }}>
            Drop here or <span style={{ color:accent }}>browse</span>
          </span>}
    </div>
  );
}

function Spinner() {
  return <span style={{ width:17, height:17, border:"2px solid rgba(255,255,255,.3)",
    borderTopColor:"#fff", borderRadius:"50%", animation:"spin .7s linear infinite",
    display:"inline-block" }} />;
}

// ─── SCAN ANIMATION ───────────────────────────────────────────────────────────
function ScanAnim() {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
      gap:18, padding:"44px 0", animation:"fadeIn .3s ease" }}>
      <div style={{ width:108, height:108, position:"relative",
        display:"flex", alignItems:"center", justifyContent:"center" }}>
        {[0,1,2].map(i => (
          <div key={i} style={{ position:"absolute", inset:0, borderRadius:"50%",
            border:"1px solid var(--teal)", animation:`ring 2s ease-out ${i*.55}s infinite` }} />
        ))}
        <div style={{ width:52, height:52, borderRadius:"50%", color:"var(--teal)",
          background:"radial-gradient(circle,rgba(0,212,200,.17),transparent)",
          display:"flex", alignItems:"center", justifyContent:"center" }}>
          <Icon.Brain />
        </div>
        <div style={{ position:"absolute", left:0, right:0, height:1,
          background:"linear-gradient(90deg,transparent,var(--teal),transparent)",
          animation:"scanln 1.4s linear infinite", boxShadow:"0 0 8px var(--teal)" }} />
      </div>
      <div style={{ textAlign:"center" }}>
        <div style={{ fontWeight:700, marginBottom:5 }}>Analyzing Neural Patterns</div>
        <div style={{ color:"var(--t2)", fontSize:13, fontFamily:"var(--fm)" }}>
          3D-CNN inference · GradCAM heatmap…
        </div>
      </div>
    </div>
  );
}

// ─── RESULT PANEL ─────────────────────────────────────────────────────────────
function ResultPanel({ result }) {
  const { curr, past, mode } = result;
  const isCompare = mode === "compare" && past;
  const delta = isCompare ? (curr.probability - past.probability) * 100 : 0;
  const recs = getRecommendations(curr.probability, isCompare, delta);

  return (
    <div style={{ animation:"fadeUp .45s ease" }}>
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:22 }}>
        <div style={{ width:6, height:6, borderRadius:"50%", background:"var(--grn)",
          boxShadow:"0 0 8px var(--grn)", animation:"blink 2.5s ease infinite" }} />
        <span style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t2)", letterSpacing:2 }}>
          ANALYSIS COMPLETE
        </span>
      </div>
      {isCompare
        ? <CompareResult curr={curr} past={past} delta={delta} />
        : <SingleResult data={curr} />}
      <RecommendationsPanel recs={recs} />
    </div>
  );
}

// ─── SINGLE RESULT ────────────────────────────────────────────────────────────
function SingleResult({ data }) {
  const prob = data.probability ?? 0;
  const pct  = (prob * 100).toFixed(1);
  // Prefer color from backend (risk_color), fallback to computed
  const color = data.risk_color || riskMeta(prob).color;

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:13, marginBottom:22 }}>
      {/* Probability hero */}
      <div style={{ gridColumn:"span 2", background:"var(--panel)", border:"1px solid var(--b)",
        borderRadius:16, padding:"22px 26px" }}>
        <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between" }}>
          <div>
            <div style={{ fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)",
              letterSpacing:1.5, marginBottom:10 }}>STROKE PROBABILITY — MODEL OUTPUT</div>
            <div style={{ fontSize:66, fontWeight:800, lineHeight:1, color,
              fontVariantNumeric:"tabular-nums" }}>
              {pct}<span style={{ fontSize:28 }}>%</span>
            </div>
            <div style={{ marginTop:7, fontSize:12, color:"var(--t3)", fontFamily:"var(--fm)" }}>
              Raw sigmoid: {prob.toFixed(6)} · Temperature-calibrated
            </div>
          </div>
          <RiskBadge color={color} label={data.risk_level} />
        </div>
        <ProbBar pct={parseFloat(pct)} color={color} />
        <div style={{ display:"flex", gap:14, marginTop:9, fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)" }}>
          <span><span style={{ color:"var(--grn)" }}>●</span> Low &lt;9%</span>
          <span><span style={{ color:"var(--amb)" }}>●</span> Moderate 9–24%</span>
          <span><span style={{ color:"var(--red)" }}>●</span> High ≥25%</span>
        </div>
      </div>

      {/* GradCAM */}
      {data.heatmap_b64 && <HeatmapCard b64={data.heatmap_b64} />}

      {/* Scan details */}
      <InfoCard rows={[
        ["File ID",      data.id],
        ["Probability",  `${pct}%`],
        ["Risk Level",   data.risk_level],
        ["Backend Risk", data.risk_color ? "Color confirmed ✓" : "Computed locally"],
        ["Model",        "StrokeCNN3D + GradCAM"],
        ["Calibration",  "Temperature scaling (T)"],
      ]} />
    </div>
  );
}

// ─── COMPARE RESULT ───────────────────────────────────────────────────────────
function CompareResult({ curr, past, delta }) {
  const improving = delta < -2;
  const worsening = delta > 2;
  const tc = improving ? "var(--grn)" : worsening ? "var(--red)" : "var(--amb)";
  const tl = improving ? "▼ Improving" : worsening ? "▲ Worsening" : "═ Stable";

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:13, marginBottom:22 }}>
      {/* Delta banner */}
      <div style={{ gridColumn:"span 2", borderRadius:16, padding:"18px 24px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
        background: improving ? "rgba(46,213,115,.05)" : worsening ? "rgba(255,71,87,.05)" : "rgba(255,165,2,.05)",
        border:`1px solid ${improving ? "rgba(46,213,115,.26)" : worsening ? "rgba(255,71,87,.26)" : "rgba(255,165,2,.22)"}` }}>
        <div>
          <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)", letterSpacing:1.5, marginBottom:6 }}>
            PROGRESSION DELTA (PAST → CURRENT)
          </div>
          <div style={{ fontSize:44, fontWeight:800, color:tc, lineHeight:1 }}>
            {delta >= 0 ? "+" : ""}{delta.toFixed(1)}%
          </div>
          <div style={{ fontSize:12, color:"var(--t2)", marginTop:6, fontFamily:"var(--fm)" }}>
            {past.probability.toFixed(4)} → {curr.probability.toFixed(4)} raw probability
          </div>
        </div>
        <div style={{ padding:"10px 24px", borderRadius:30, fontWeight:700, fontSize:14,
          background:`${tc}17`, border:`1px solid ${tc}33`, color:tc }}>{tl}</div>
      </div>

      <ScanMiniCard data={past} label="PAST SCAN"    accent="#8b5cf6" />
      <ScanMiniCard data={curr} label="CURRENT SCAN" accent="var(--teal)" />

      {past.heatmap_b64 && <HeatmapCard b64={past.heatmap_b64}    label="Past — GradCAM" />}
      {curr.heatmap_b64 && <HeatmapCard b64={curr.heatmap_b64} label="Current — GradCAM" />}
    </div>
  );
}

// ─── SHARED SMALL COMPONENTS ──────────────────────────────────────────────────
function RiskBadge({ color, label }) {
  return (
    <div style={{ padding:"7px 16px", borderRadius:30,
      background:`${color}17`, border:`1px solid ${color}36`,
      color, fontWeight:700, fontSize:13, letterSpacing:.4, flexShrink:0 }}>
      {label}
    </div>
  );
}

function ProbBar({ pct, color }) {
  return (
    <div style={{ marginTop:17, height:7, background:"var(--deep)", borderRadius:99 }}>
      <div style={{ height:"100%", width:`${pct}%`,
        background:`linear-gradient(90deg,${color}66,${color})`,
        borderRadius:99, boxShadow:`0 0 10px ${color}88`, transition:"width 1s ease" }} />
    </div>
  );
}

function ScanMiniCard({ data, label, accent }) {
  const prob  = data.probability ?? 0;
  const pct   = (prob * 100).toFixed(1);
  const color = data.risk_color || riskMeta(prob).color;
  return (
    <div style={{ background:"var(--panel)", border:"1px solid var(--b)", borderRadius:16, padding:22 }}>
      <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)", letterSpacing:1.5, marginBottom:10 }}>
        {label}
      </div>
      <div style={{ fontSize:50, fontWeight:800, color, lineHeight:1, marginBottom:10 }}>{pct}%</div>
      <RiskBadge color={color} label={data.risk_level} />
      <ProbBar pct={parseFloat(pct)} color={color} />
    </div>
  );
}

function HeatmapCard({ b64, label }) {
  return (
    <div style={{ background:"var(--panel)", border:"1px solid var(--b)", borderRadius:16, padding:20 }}>
      <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)",
        letterSpacing:1.5, marginBottom:12 }}>{label || "GRADCAM HEATMAP"}</div>
      <img src={`data:image/png;base64,${b64}`} alt="GradCAM"
        style={{ width:"100%", borderRadius:10, display:"block" }} />
      <div style={{ marginTop:8, fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)" }}>
        Hot regions → areas driving the prediction
      </div>
    </div>
  );
}

function InfoCard({ rows }) {
  return (
    <div style={{ background:"var(--panel)", border:"1px solid var(--b)", borderRadius:16, padding:22 }}>
      <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)",
        letterSpacing:1.5, marginBottom:14 }}>SCAN DETAILS</div>
      {rows.map(([k,v]) => (
        <div key={k} style={{ display:"flex", justifyContent:"space-between",
          padding:"7px 0", borderBottom:"1px solid var(--b)", fontSize:13 }}>
          <span style={{ color:"var(--t2)" }}>{k}</span>
          <span style={{ fontFamily:"var(--fm)", color:"var(--t1)", textAlign:"right",
            maxWidth:200, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

// ─── RECOMMENDATIONS ──────────────────────────────────────────────────────────
const PM = {
  URGENT:   { color:"#ff4757", bg:"rgba(255,71,87,.08)",  bdr:"rgba(255,71,87,.2)",  tag:"URGENT"   },
  HIGH:     { color:"#ff8c00", bg:"rgba(255,140,0,.08)",  bdr:"rgba(255,140,0,.2)",  tag:"HIGH"     },
  MODERATE: { color:"#ffa502", bg:"rgba(255,165,2,.07)",  bdr:"rgba(255,165,2,.18)", tag:"MODERATE" },
  LOW:      { color:"#2ed573", bg:"rgba(46,213,115,.06)", bdr:"rgba(46,213,115,.17)",tag:"ROUTINE"  },
};

function RecommendationsPanel({ recs }) {
  return (
    <div style={{ marginBottom:24 }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:16 }}>
        <div style={{ width:3, height:22, borderRadius:2,
          background:"linear-gradient(180deg,var(--teal),var(--blue))" }} />
        <h2 style={{ fontSize:18, fontWeight:700 }}>Clinical Recommendations</h2>
        <span style={{ fontSize:10, fontFamily:"var(--fm)", color:"var(--t3)",
          padding:"2px 8px", border:"1px solid var(--b)", borderRadius:5 }}>
          AI-ASSISTED · NOT A SUBSTITUTE FOR CLINICAL JUDGMENT
        </span>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:11 }}>
        {recs.map((r, i) => {
          const m = PM[r.priority] || PM.LOW;
          return (
            <div key={i} style={{ background:m.bg, border:`1px solid ${m.bdr}`,
              borderRadius:13, padding:"16px 18px",
              animation:`fadeUp .4s ease ${i*.07}s both` }}>
              <div style={{ display:"flex", alignItems:"flex-start", gap:11 }}>
                <span style={{ fontSize:20, flexShrink:0, lineHeight:1.2 }}>{r.icon}</span>
                <div>
                  <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:5 }}>
                    <span style={{ fontSize:13, fontWeight:700 }}>{r.title}</span>
                    <span style={{ padding:"2px 6px", borderRadius:4, fontSize:10, fontWeight:700,
                      fontFamily:"var(--fm)", background:`${m.color}1e`, color:m.color,
                      letterSpacing:.4 }}>{m.tag}</span>
                  </div>
                  <p style={{ fontSize:13, color:"var(--t2)", lineHeight:1.55 }}>{r.body}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop:12, padding:"9px 14px",
        background:"rgba(74,158,255,.05)", border:"1px solid rgba(74,158,255,.14)",
        borderRadius:9, fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)" }}>
        ⚠ AI-generated recommendations — must be reviewed by a licensed clinician before any clinical action.
        Thresholds: High ≥20% · Moderate 7–20% · Low &lt;7%
      </div>
    </div>
  );
}

// ─── TRENDS VIEW ──────────────────────────────────────────────────────────────
function TrendsView({ history }) {
  // Build chronological scan list from history (newest first in history array)
  const scans = [...history]
    .reverse()
    .filter(h => h.curr)
    .map((h, idx) => ({
      idx:   idx + 1,
      prob:  h.curr.probability,
      risk:  h.curr.risk_level,
      color: h.curr.risk_color || riskMeta(h.curr.probability).color,
      label: h.filename,
      ts:    h.ts,
    }));

  if (scans.length < 2) {
    return (
      <div style={{ textAlign:"center", padding:"80px 0", animation:"fadeIn .4s ease" }}>
        <div style={{ width:64, height:64, borderRadius:"50%", background:"var(--panel)",
          border:"1px solid var(--b)", display:"flex", alignItems:"center", justifyContent:"center",
          margin:"0 auto 20px", color:"var(--t3)" }}>
          <Icon.Chart />
        </div>
        <div style={{ fontWeight:700, color:"var(--t2)", marginBottom:8 }}>Not enough data</div>
        <div style={{ color:"var(--t3)", fontSize:13 }}>
          Run at least 2 scans in the Diagnose tab to see the risk trend chart.
        </div>
      </div>
    );
  }

  // SVG chart dimensions
  const W=760, H=260, P={ t:24, r:24, b:44, l:54 };
  const gW = W-P.l-P.r, gH = H-P.t-P.b;
  const probs = scans.map(s => s.prob);
  const maxP  = Math.max(...probs, 0.32);
  const minP  = Math.min(...probs, 0);
  const xOf   = i => P.l + (i/(scans.length-1))*gW;
  const yOf   = p => P.t + gH - ((p-minP)/(maxP-minP||1))*gH;
  const pts   = scans.map((s,i) => [xOf(i), yOf(s.prob)]);

  // Smooth bezier line
  const linePath = pts.reduce((acc,[x,y],i) => {
    if (i===0) return `M ${x} ${y}`;
    const [px,py]=pts[i-1], cx=(px+x)/2;
    return `${acc} C ${cx} ${py},${cx} ${y},${x} ${y}`;
  },"");
  const areaPath = `${linePath} L ${pts[pts.length-1][0]} ${P.t+gH} L ${P.l} ${P.t+gH} Z`;

  // Risk threshold y-positions
  const y25 = yOf(0.25), y09 = yOf(0.09);

  const last = scans[scans.length-1];
  const prev = scans[scans.length-2];
  const sessionDelta = ((last.prob - prev.prob)*100).toFixed(1);

  return (
    <div style={{ animation:"fadeUp .4s ease" }}>
      <div style={{ marginBottom:24 }}>
        <h1 style={{ fontSize:28, fontWeight:800, letterSpacing:"-.5px", marginBottom:6 }}>
          Risk{" "}
          <span style={{ background:"linear-gradient(90deg,var(--teal),var(--blue))",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Trend</span>
        </h1>
        <p style={{ color:"var(--t2)", fontSize:13 }}>
          Stroke probability across {scans.length} scans — chronological order
        </p>
      </div>

      {/* Stat cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:11, marginBottom:18 }}>
        {[
          ["Latest Risk",     `${(last.prob*100).toFixed(1)}%`,    last.color],
          ["Session Change",  `${parseFloat(sessionDelta)>=0?"+":""}${sessionDelta}%`,
                              parseFloat(sessionDelta)<0?"var(--grn)":"var(--red)"],
          ["Scans Analyzed",  scans.length,                        "var(--blue)"],
          ["Risk Category",   last.risk,                           last.color],
        ].map(([lbl,val,clr]) => (
          <div key={lbl} style={{ background:"var(--panel)", border:"1px solid var(--b)",
            borderRadius:14, padding:"15px 18px" }}>
            <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)",
              letterSpacing:1.2, marginBottom:7 }}>{lbl}</div>
            <div style={{ fontSize:24, fontWeight:800, color:clr }}>{val}</div>
          </div>
        ))}
      </div>

      {/* SVG chart */}
      <div style={{ background:"var(--panel)", border:"1px solid var(--b)",
        borderRadius:16, padding:"20px 22px", marginBottom:18, overflowX:"auto" }}>
        <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)",
          letterSpacing:1.5, marginBottom:14 }}>PROBABILITY OVER TIME</div>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
          style={{ display:"block", maxWidth:"100%", height:"auto", overflow:"visible" }}>
          <defs>
            <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#00d4c8" stopOpacity=".17" />
              <stop offset="100%" stopColor="#00d4c8" stopOpacity="0"   />
            </linearGradient>
          </defs>

          {/* Zone fills */}
          <rect x={P.l} y={P.t}  width={gW} height={Math.max(0,y25-P.t)}   fill="rgba(255,71,87,.04)"  />
          <rect x={P.l} y={y25}  width={gW} height={Math.max(0,y09-y25)}   fill="rgba(255,165,2,.04)" />
          <rect x={P.l} y={y09}  width={gW} height={Math.max(0,P.t+gH-y09)} fill="rgba(46,213,115,.03)"/>

          {/* Zone labels */}
          {[[y25-9,"#ff4757","HIGH ≥20%"],[( y25+y09)/2-6,"#ffa502","MODERATE"],[y09+9,"#2ed573","LOW <5%"]]
            .map(([y,c,t]) => (
            <text key={t} x={P.l+5} y={y} fill={c} fontSize={9} fontFamily="var(--fm)" opacity={.75}>{t}</text>
          ))}

          {/* Threshold dashes */}
          {[[y25,"#ff4757"],[y09,"#ffa502"]].map(([y,c],i) => (
            <line key={i} x1={P.l} y1={y} x2={P.l+gW} y2={y}
              stroke={c} strokeWidth={1} strokeDasharray="4 4" opacity={.38} />
          ))}

          {/* Y-axis */}
          {[0,.1,.2,.3,.5,.75,1].filter(v=>v>=minP-.05&&v<=maxP+.05).map(v => (
            <text key={v} x={P.l-7} y={yOf(v)+4} fill="var(--t3)" fontSize={10}
              fontFamily="var(--fm)" textAnchor="end">{(v*100).toFixed(0)}%</text>
          ))}

          {/* Area + line */}
          <path d={areaPath} fill="url(#ag)" />
          <path d={linePath} fill="none" stroke="var(--teal)" strokeWidth={2.5}
            strokeLinecap="round" strokeLinejoin="round" />

          {/* Dots */}
          {pts.map(([x,y],i) => (
            <g key={i}>
              <circle cx={x} cy={y} r={9} fill={scans[i].color} opacity={.12} />
              <circle cx={x} cy={y} r={5.5} fill={scans[i].color} stroke="var(--deep)" strokeWidth={2} />
            </g>
          ))}

          {/* X labels */}
          {scans.map((_,i) => (
            <text key={i} x={xOf(i)} y={H-10} fill="var(--t3)" fontSize={9}
              fontFamily="var(--fm)" textAnchor="middle">#{i+1}</text>
          ))}
        </svg>
      </div>

      {/* Scan log table */}
      <div style={{ background:"var(--panel)", border:"1px solid var(--b)", borderRadius:16, padding:"20px 22px" }}>
        <div style={{ fontSize:11, fontFamily:"var(--fm)", color:"var(--t3)", letterSpacing:1.5, marginBottom:13 }}>
          SCAN LOG
        </div>
        {/* Header */}
        <div style={{ display:"grid", gridTemplateColumns:"36px 1fr 90px 130px 130px",
          gap:"0 10px", fontSize:10, fontFamily:"var(--fm)", color:"var(--t3)",
          borderBottom:"1px solid var(--b)", paddingBottom:7, marginBottom:6, letterSpacing:.8 }}>
          {["#","FILE","PROB","RISK","TIMESTAMP"].map(h => <span key={h}>{h}</span>)}
        </div>
        {scans.map((s,i) => {
          const prev = i > 0 ? scans[i-1].prob : null;
          const d = prev !== null ? ((s.prob-prev)*100).toFixed(1) : null;
          return (
            <div key={i} style={{ display:"grid",
              gridTemplateColumns:"36px 1fr 90px 130px 130px",
              gap:"0 10px", padding:"8px 0", borderBottom:"1px solid var(--b)",
              fontSize:13, alignItems:"center", animation:`fadeUp .35s ease ${i*.04}s both` }}>
              <span style={{ fontFamily:"var(--fm)", color:"var(--t3)", fontSize:12 }}>#{s.idx}</span>
              <span style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                {s.label}
              </span>
              <span style={{ fontFamily:"var(--fm)", color:s.color, fontWeight:700 }}>
                {(s.prob*100).toFixed(1)}%
                {d !== null && (
                  <span style={{ marginLeft:4, fontSize:10,
                    color: parseFloat(d)<0 ? "var(--grn)" : "var(--red)" }}>
                    ({parseFloat(d)>=0?"+":""}{d})
                  </span>
                )}
              </span>
              <span style={{ padding:"3px 9px", borderRadius:20,
                background:`${s.color}11`, border:`1px solid ${s.color}26`,
                color:s.color, fontSize:11, fontWeight:700, fontFamily:"var(--fm)",
                display:"inline-block" }}>{s.risk}</span>
              <span style={{ fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)" }}>{s.ts}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── HISTORY VIEW ─────────────────────────────────────────────────────────────
function HistoryView({ history }) {
  if (!history.length) {
    return (
      <div style={{ textAlign:"center", padding:"80px 0", animation:"fadeIn .4s ease" }}>
        <div style={{ width:64, height:64, borderRadius:"50%", background:"var(--panel)",
          border:"1px solid var(--b)", display:"flex", alignItems:"center", justifyContent:"center",
          margin:"0 auto 20px", color:"var(--t3)" }}>
          <Icon.History />
        </div>
        <div style={{ fontWeight:700, color:"var(--t2)", marginBottom:8 }}>No diagnoses yet</div>
        <div style={{ color:"var(--t3)", fontSize:13 }}>Run a scan in the Diagnose tab to begin.</div>
      </div>
    );
  }
  return (
    <div style={{ animation:"fadeUp .4s ease" }}>
      <div style={{ marginBottom:22 }}>
        <h1 style={{ fontSize:28, fontWeight:800, marginBottom:6 }}>
          Diagnosis{" "}
          <span style={{ background:"linear-gradient(90deg,var(--teal),var(--blue))",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>History</span>
        </h1>
        <p style={{ color:"var(--t2)", fontSize:13 }}>
          {history.length} session{history.length!==1?"s":""} this session
        </p>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:9 }}>
        {history.map((h,i) => {
          const d   = h.curr || h;
          const prob = d.probability ?? 0;
          const color = d.risk_color || riskMeta(prob).color;
          const pct  = (prob*100).toFixed(1);
          return (
            <div key={i} style={{ background:"var(--panel)", border:"1px solid var(--b)",
              borderRadius:13, padding:"15px 21px",
              display:"flex", alignItems:"center", gap:17,
              animation:`fadeUp .4s ease ${i*.05}s both` }}>
              <div style={{ width:44, height:44, borderRadius:11, flexShrink:0,
                background:`${color}11`, border:`1px solid ${color}26`,
                display:"flex", alignItems:"center", justifyContent:"center",
                color, fontWeight:800, fontSize:13, fontFamily:"var(--fm)" }}>
                {pct}%
              </div>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ fontWeight:600, fontSize:14, marginBottom:3,
                  overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                  {h.filename || d.id}
                </div>
                <div style={{ fontSize:11, color:"var(--t3)", fontFamily:"var(--fm)" }}>
                  {h.ts} · {h.mode==="compare"?"Comparison":"Single scan"}
                  {h.pastFilename ? ` · vs ${h.pastFilename}` : ""}
                </div>
              </div>
              <div style={{ padding:"4px 12px", borderRadius:20,
                background:`${color}10`, border:`1px solid ${color}24`,
                color, fontSize:11, fontWeight:700, fontFamily:"var(--fm)", flexShrink:0 }}>
                {d.risk_level}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── ROOT ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(null);
  return (
    <>
      <G />
      <div style={{ position:"relative", zIndex:1, minHeight:"100vh" }}>
        {user
          ? <Dashboard user={user} onLogout={() => setUser(null)} />
          : <AuthPage onAuth={setUser} />}
      </div>
    </>
  );
}
