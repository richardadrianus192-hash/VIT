// frontend/src/TrainingPanel.jsx
// VIT Sports Intelligence — v2.4.0  Training Pipeline

import { useEffect, useRef, useState } from 'react'
import { API_KEY } from './api'

const API_BASE = import.meta.env.VITE_API_URL || ''

// ── Styles ────────────────────────────────────────────────────────────
const card  = { background:'#fff', border:'1px solid #e2e8f0', borderRadius:12, padding:'20px 24px', marginBottom:20, boxShadow:'0 2px 8px rgba(15,23,42,0.06)' }
const title = { fontSize:'1rem', fontWeight:700, color:'#0f172a', marginBottom:14, marginTop:0 }
const lbl   = { display:'block', fontSize:'0.78rem', fontWeight:600, color:'#475569', marginBottom:4 }
const inp   = { width:'100%', padding:'8px 12px', border:'1px solid #cbd5e1', borderRadius:8, fontSize:'0.9rem', background:'#f8fafc', outline:'none' }
const btnP  = { background:'linear-gradient(135deg,#0ea5e9,#6366f1)', color:'#fff', border:'none', borderRadius:8, padding:'9px 20px', fontWeight:600, fontSize:'0.88rem', cursor:'pointer' }
const btnS  = { background:'#f1f5f9', color:'#334155', border:'1px solid #e2e8f0', borderRadius:8, padding:'9px 20px', fontWeight:600, fontSize:'0.88rem', cursor:'pointer' }
const pill  = c => ({ display:'inline-block', padding:'2px 10px', borderRadius:99, fontSize:'0.75rem', fontWeight:700,
  background: c==='green'?'#dcfce7':c==='red'?'#fee2e2':c==='yellow'?'#fef9c3':c==='blue'?'#dbeafe':'#f1f5f9',
  color:      c==='green'?'#15803d':c==='red'?'#b91c1c':c==='yellow'?'#92400e':c==='blue'?'#1d4ed8':'#64748b' })

function apiFetch(path, opts={}) {
  return fetch(`${API_BASE}${path}`, { headers:{'Content-Type':'application/json','x-api-key':API_KEY}, ...opts }).then(r => {
    if (!r.ok) return r.text().then(t => { throw new Error(t || r.statusText) })
    return r.json()
  })
}

const LEAGUES = ['premier_league','la_liga','bundesliga','serie_a','ligue_1']

export default function TrainingPanel({ apiKey }) {
  const key = apiKey || API_KEY

  // Config
  const [config, setConfig] = useState({
    leagues: LEAGUES, date_from:'2023-01-01', date_to:'2025-12-31',
    validation_split:0.20, max_epochs:100, note:''
  })
  const [starting, setStarting] = useState(false)
  const [startErr, setStartErr] = useState('')

  // Active job streaming
  const [jobId, setJobId]       = useState(null)
  const [jobStatus, setJobStatus] = useState('idle')
  const [events, setEvents]     = useState([])
  const [progress, setProgress] = useState({ current:0, total:0 })
  const [modelResults, setModelResults] = useState([])
  const esRef = useRef(null)
  const logRef = useRef(null)

  // Jobs list
  const [jobs, setJobs]         = useState([])
  const [jobsLoading, setJobsL] = useState(false)

  // Compare
  const [cmpA, setCmpA] = useState('')
  const [cmpB, setCmpB] = useState('')
  const [comparison, setComparison] = useState(null)
  const [cmpLoading, setCmpL] = useState(false)
  const [cmpErr, setCmpErr] = useState('')

  // Promote
  const [promoting, setPromoting] = useState(false)
  const [promoteMsg, setPromoteMsg] = useState('')

  useEffect(() => { loadJobs(); return () => esRef.current?.close() }, [])
  useEffect(() => { logRef.current?.scrollIntoView({ behavior:'smooth' }) }, [events])

  async function loadJobs() {
    setJobsL(true)
    try { const r = await apiFetch(`/training/jobs?api_key=${encodeURIComponent(key)}`); setJobs(r.jobs || []) }
    catch(e) { console.error(e) } finally { setJobsL(false) }
  }

  async function startTraining() {
    setStarting(true); setStartErr(''); setEvents([]); setModelResults([]); setProgress({current:0,total:0})
    try {
      const r = await apiFetch(`/training/start?api_key=${encodeURIComponent(key)}`, {
        method:'POST', body: JSON.stringify(config)
      })
      setJobId(r.job_id); setJobStatus('queued')
      streamProgress(r.job_id)
      await loadJobs()
    } catch(e) { setStartErr(e.message) } finally { setStarting(false) }
  }

  function streamProgress(jid) {
    esRef.current?.close()
    const es = new EventSource(`${API_BASE}/training/progress/${jid}?api_key=${encodeURIComponent(key)}`)
    esRef.current = es
    es.onmessage = (e) => {
      const d = JSON.parse(e.data)
      if (d.type === 'heartbeat')    { setProgress({ current: d.current, total: d.total }); setJobStatus(d.status) }
      if (d.type === 'model_start')  { setEvents(ev => [...ev, `[${d.index}/${d.total}] Training ${d.model}…`]) }
      if (d.type === 'model_done')   { setModelResults(mr => [...mr, { name: d.model, accuracy: d.accuracy, elapsed: d.elapsed_s, ok:true }]) }
      if (d.type === 'model_error')  { setModelResults(mr => [...mr, { name: d.model, error: d.error, ok:false }]) }
      if (d.type === 'done')         { setJobStatus('completed'); setEvents(ev => [...ev, `✅ Complete — avg accuracy: ${(d.summary?.avg_accuracy*100||0).toFixed(1)}%`]); es.close(); loadJobs() }
      if (d.type === 'stream_end')   { es.close(); loadJobs() }
    }
    es.onerror = () => { setJobStatus('error'); es.close() }
  }

  async function compare() {
    if (!cmpA || !cmpB) { setCmpErr('Select both versions'); return }
    setCmpL(true); setCmpErr(''); setComparison(null)
    try { setComparison(await apiFetch(`/training/compare?job_id_a=${cmpA}&job_id_b=${cmpB}&api_key=${encodeURIComponent(key)}`)) }
    catch(e) { setCmpErr(e.message) } finally { setCmpL(false) }
  }

  async function promote(jid) {
    setPromoting(true); setPromoteMsg('')
    try {
      const r = await apiFetch(`/training/promote?api_key=${encodeURIComponent(key)}`, {
        method:'POST', body: JSON.stringify({ job_id: jid, reason:'Manually promoted from UI' })
      })
      setPromoteMsg(`✅ Version ${jid.slice(0,8)} promoted to production`)
      await loadJobs()
    } catch(e) { setPromoteMsg(`❌ ${e.message}`) }
    finally { setPromoting(false) }
  }

  const completedJobs = jobs.filter(j => j.status === 'completed')
  const pct = progress.total > 0 ? Math.round(progress.current / progress.total * 100) : 0

  return (
    <div style={{ maxWidth:1000, margin:'0 auto' }}>

      {/* ── Start Training ─────────────────────────────────────── */}
      <div style={card}>
        <h3 style={title}>🧠 Start Training Run</h3>
        <p style={{ fontSize:'0.85rem', color:'#64748b', marginTop:-8, marginBottom:16 }}>
          Trains all loaded models on historical match data. Progress streams in real-time below.
        </p>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))', gap:14, marginBottom:16 }}>
          {[['date_from','From Date','date'],['date_to','To Date','date']].map(([k,l,t]) => (
            <div key={k}><label style={lbl}>{l}</label>
              <input style={inp} type={t} value={config[k]} onChange={e => setConfig(c=>({...c,[k]:e.target.value}))} /></div>
          ))}
          <div><label style={lbl}>Validation Split</label>
            <input style={inp} type="number" step="0.05" min="0.1" max="0.4"
              value={config.validation_split} onChange={e=>setConfig(c=>({...c,validation_split:parseFloat(e.target.value)}))} /></div>
          <div><label style={lbl}>Max Epochs</label>
            <input style={inp} type="number" min="10" max="500"
              value={config.max_epochs} onChange={e=>setConfig(c=>({...c,max_epochs:parseInt(e.target.value)}))} /></div>
        </div>
        <div style={{ marginBottom:16 }}><label style={lbl}>Run Note (optional)</label>
          <input style={inp} type="text" placeholder="e.g. Added new feature set"
            value={config.note} onChange={e=>setConfig(c=>({...c,note:e.target.value}))} /></div>

        {startErr && <div style={{ marginBottom:12, padding:'8px 12px', background:'#fee2e2', borderRadius:8, color:'#b91c1c', fontSize:'0.85rem' }}>{startErr}</div>}

        <div style={{ display:'flex', gap:12 }}>
          <button style={btnP} onClick={startTraining} disabled={starting || jobStatus==='running'}>
            {starting ? 'Starting…' : jobStatus==='running' ? '⏳ Training in progress…' : '▶ Start Training'}
          </button>
        </div>
      </div>

      {/* ── Live Progress ─────────────────────────────────────── */}
      {(jobStatus !== 'idle' || events.length > 0) && (
        <div style={card}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12 }}>
            <h3 style={{...title, marginBottom:0}}>
              ⚡ Live Progress
              {jobId && <span style={{ marginLeft:10, fontFamily:'monospace', fontSize:'0.78rem', color:'#64748b' }}>#{jobId}</span>}
            </h3>
            <span style={pill(jobStatus==='completed'?'green':jobStatus==='running'?'blue':jobStatus==='error'?'red':'gray')}>
              {jobStatus}
            </span>
          </div>

          {progress.total > 0 && (
            <div style={{ marginBottom:14 }}>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.82rem', color:'#64748b', marginBottom:5 }}>
                <span>Models trained</span><span>{progress.current}/{progress.total} ({pct}%)</span>
              </div>
              <div style={{ background:'#e2e8f0', borderRadius:99, height:10, overflow:'hidden' }}>
                <div style={{ width:`${pct}%`, background:'linear-gradient(90deg,#10b981,#0ea5e9)', height:'100%', borderRadius:99, transition:'width 0.5s' }} />
              </div>
            </div>
          )}

          {/* Model results grid */}
          {modelResults.length > 0 && (
            <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:8, marginBottom:14 }}>
              {modelResults.map((m,i) => (
                <div key={i} style={{ padding:'8px 12px', background: m.ok?'#f0fdf4':'#fff1f2', borderRadius:8, border:`1px solid ${m.ok?'#86efac':'#fca5a5'}`, fontSize:'0.82rem' }}>
                  <div style={{ fontWeight:700, marginBottom:2 }}>{m.name}</div>
                  {m.ok
                    ? <><span style={{ color:'#15803d' }}>{(m.accuracy*100).toFixed(1)}% acc</span> <span style={{ color:'#94a3b8' }}>{m.elapsed}s</span></>
                    : <span style={{ color:'#b91c1c' }}>✕ {m.error?.slice(0,40)}</span>}
                </div>
              ))}
            </div>
          )}

          <div style={{ background:'#f8fafc', borderRadius:8, padding:'10px 14px', maxHeight:120, overflowY:'auto', fontSize:'0.8rem', fontFamily:'monospace', color:'#475569' }}>
            {events.map((e,i) => <div key={i}>{e}</div>)}
            {events.length === 0 && <span style={{ color:'#94a3b8' }}>Waiting for events…</span>}
            <div ref={logRef} />
          </div>
        </div>
      )}

      {/* ── Training History ───────────────────────────────────── */}
      <div style={card}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14 }}>
          <h3 style={{...title, marginBottom:0}}>📋 Training History</h3>
          <button style={btnS} onClick={loadJobs} disabled={jobsLoading}>{jobsLoading?'Loading…':'↻ Refresh'}</button>
        </div>

        {jobs.length === 0
          ? <p style={{ color:'#94a3b8', fontSize:'0.88rem', margin:0 }}>No training runs yet.</p>
          : <div style={{ overflowX:'auto' }}>
              <table style={{ width:'100%', borderCollapse:'collapse' }}>
                <thead><tr style={{ background:'#f8fafc', borderBottom:'2px solid #e2e8f0' }}>
                  {['Job ID','Status','Avg Accuracy','Models','Created',''].map(h =>
                    <th key={h} style={{ padding:'7px 12px', textAlign:'left', fontSize:'0.76rem', fontWeight:700, color:'#475569' }}>{h}</th>
                  )}
                </tr></thead>
                <tbody>
                  {jobs.map(j => (
                    <tr key={j.job_id} style={{ borderBottom:'1px solid #f1f5f9' }}>
                      <td style={{ padding:'7px 12px', fontFamily:'monospace', fontSize:'0.82rem', color:'#64748b' }}>
                        {j.job_id.slice(0,8)}
                        {j.is_production && <span style={{ marginLeft:6, ...pill('green') }}>★ PROD</span>}
                      </td>
                      <td style={{ padding:'7px 12px' }}><span style={pill(j.status==='completed'?'green':j.status==='running'?'blue':'red')}>{j.status}</span></td>
                      <td style={{ padding:'7px 12px', fontSize:'0.85rem', fontWeight:600 }}>
                        {j.avg_accuracy ? `${(j.avg_accuracy*100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding:'7px 12px', fontSize:'0.82rem' }}>{j.models_trained || '—'}</td>
                      <td style={{ padding:'7px 12px', fontSize:'0.8rem', color:'#94a3b8' }}>
                        {j.created_at ? new Date(j.created_at).toLocaleString('en-GB',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}) : '—'}
                      </td>
                      <td style={{ padding:'7px 12px' }}>
                        {j.status==='completed' && !j.is_production && (
                          <button style={{ ...btnP, padding:'5px 12px', fontSize:'0.78rem' }}
                            onClick={()=>promote(j.job_id)} disabled={promoting}>
                            {promoting?'…':'⬆ Promote'}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        }
        {promoteMsg && <div style={{ marginTop:10, fontSize:'0.85rem', color: promoteMsg.startsWith('✅')?'#15803d':'#b91c1c', fontWeight:600 }}>{promoteMsg}</div>}
      </div>
      {/* ── Models Information with Child Models ─────────────── */}
      <div style={card}>
        <h3 style={title}>🤖 Model Architecture & Child Networks</h3>
        <p style={{ fontSize:'0.85rem', color:'#64748b', marginTop:-8, marginBottom:16 }}>
          Transparent view of all models, their types, and internal child networks.
        </p>
        
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(250px,1fr))', gap:16 }}>
          {/* Placeholder - will load from API in real component */}
          <div style={{ padding:12, background:'#f8fafc', borderRadius:10, border:'1px solid #e2e8f0' }}>
            <div style={{ fontSize:'0.9rem', fontWeight:700, marginBottom:8 }}>📊 Poisson</div>
            <div style={{ fontSize:'0.75rem', color:'#64748b', marginBottom:6 }}>Type: Goal Prediction Model</div>
            <div style={{ fontSize:'0.75rem', color:'#0f172a', fontWeight:600, marginBottom:4 }}>Child Networks:</div>
            <ul style={{ fontSize:'0.75rem', color:'#475569', margin:'4px 0 0 16px', padding:0 }}>
              <li>Poisson Distribution Generator</li>
              <li>Goal Rate Estimator</li>
              <li>Covariance Matrix</li>
            </ul>
          </div>
          
          <div style={{ padding:12, background:'#f8fafc', borderRadius:10, border:'1px solid #e2e8f0' }}>
            <div style={{ fontSize:'0.9rem', fontWeight:700, marginBottom:8 }}>🎯 XGBoost</div>
            <div style={{ fontSize:'0.75rem', color:'#64748b', marginBottom:6 }}>Type: Outcome Classifier</div>
            <div style={{ fontSize:'0.75rem', color:'#0f172a', fontWeight:600, marginBottom:4 }}>Child Networks:</div>
            <ul style={{ fontSize:'0.75rem', color:'#475569', margin:'4px 0 0 16px', padding:0 }}>
              <li>Gradient Booster (Home)</li>
              <li>Gradient Booster (Draw)</li>
              <li>Gradient Booster (Away)</li>
            </ul>
          </div>
          
          <div style={{ padding:12, background:'#f8fafc', borderRadius:10, border:'1px solid #e2e8f0' }}>
            <div style={{ fontSize:'0.9rem', fontWeight:700, marginBottom:8 }}>🎲 Monte Carlo</div>
            <div style={{ fontSize:'0.75rem', color:'#64748b', marginBottom:6 }}>Type: Probabilistic Simulator</div>
            <div style={{ fontSize:'0.75rem', color:'#0f172a', fontWeight:600, marginBottom:4 }}>Child Networks:</div>
            <ul style={{ fontSize:'0.75rem', color:'#475569', margin:'4px 0 0 16px', padding:0 }}>
              <li>Simulation Engine</li>
              <li>Probability Sampler</li>
              <li>Convergence Validator</li>
            </ul>
          </div>
        </div>
        
        <div style={{ marginTop:16, padding:12, background:'#fafaf0', borderRadius:8, fontSize:'0.8rem', color:'#475569' }}>
          <strong>💡 Note:</strong> Each model contains specialized child networks that handle specific prediction aspects. 
          Poisson, Monte Carlo, and Bayesian models provide over/under (O/U) goals predictions with specialized goal estimation child networks.
        </div>
      </div>
      {/* ── Compare Versions ──────────────────────────────────── */}
      {completedJobs.length >= 2 && (
        <div style={card}>
          <h3 style={title}>🔬 Compare Versions</h3>
          <div style={{ display:'flex', gap:12, alignItems:'flex-end', flexWrap:'wrap', marginBottom:14 }}>
            {[['Version A (baseline)', cmpA, setCmpA], ['Version B (candidate)', cmpB, setCmpB]].map(([l,v,set]) => (
              <div key={l} style={{ flex:1, minWidth:180 }}>
                <label style={lbl}>{l}</label>
                <select style={inp} value={v} onChange={e=>set(e.target.value)}>
                  <option value="">— Select —</option>
                  {completedJobs.map(j => (
                    <option key={j.job_id} value={j.job_id}>
                      {j.job_id.slice(0,8)} — {(j.avg_accuracy*100||0).toFixed(1)}% acc {j.is_production?'★':''}</option>
                  ))}
                </select>
              </div>
            ))}
            <button style={btnP} onClick={compare} disabled={cmpLoading}>
              {cmpLoading?'Comparing…':'⚡ Compare'}
            </button>
          </div>

          {cmpErr && <div style={{ padding:'8px 12px', background:'#fee2e2', borderRadius:8, color:'#b91c1c', fontSize:'0.85rem', marginBottom:12 }}>{cmpErr}</div>}

          {comparison && (
            <div>
              <div style={{ display:'flex', gap:16, marginBottom:14, flexWrap:'wrap' }}>
                <div style={{ padding:'10px 16px', background:'#f0fdf4', borderRadius:10, flex:1 }}>
                  <div style={{ fontSize:'0.78rem', color:'#64748b' }}>Version A accuracy</div>
                  <div style={{ fontSize:'1.4rem', fontWeight:800 }}>{(comparison.version_a.summary.avg_accuracy*100||0).toFixed(1)}%</div>
                </div>
                <div style={{ padding:'10px 16px', background:comparison.overall_delta>0?'#f0fdf4':'#fff1f2', borderRadius:10, flex:1 }}>
                  <div style={{ fontSize:'0.78rem', color:'#64748b' }}>Version B accuracy</div>
                  <div style={{ fontSize:'1.4rem', fontWeight:800 }}>{(comparison.version_b.summary.avg_accuracy*100||0).toFixed(1)}%</div>
                </div>
                <div style={{ padding:'10px 16px', background:'#f0f9ff', borderRadius:10, flex:1 }}>
                  <div style={{ fontSize:'0.78rem', color:'#64748b' }}>Δ Improvement</div>
                  <div style={{ fontSize:'1.4rem', fontWeight:800, color: comparison.overall_delta>0?'#15803d':'#b91c1c' }}>
                    {comparison.overall_delta>0?'+':''}{(comparison.overall_delta*100).toFixed(2)}%
                  </div>
                </div>
                <div style={{ padding:'10px 16px', background:'#fafaf0', borderRadius:10, flex:1 }}>
                  <div style={{ fontSize:'0.78rem', color:'#64748b' }}>Recommendation</div>
                  <div style={{ fontSize:'1rem', fontWeight:800, textTransform:'uppercase',
                    color: comparison.recommendation==='promote'?'#15803d':comparison.recommendation==='rollback'?'#b91c1c':'#92400e' }}>
                    {comparison.recommendation==='promote'?'✅ Promote':comparison.recommendation==='rollback'?'⬇ Rollback':'= Neutral'}
                  </div>
                </div>
              </div>

              <div style={{ overflowX:'auto' }}>
                <table style={{ width:'100%', borderCollapse:'collapse' }}>
                  <thead><tr style={{ background:'#f8fafc', borderBottom:'2px solid #e2e8f0' }}>
                    {['Model','Acc A','Acc B','Δ',''].map(h =>
                      <th key={h} style={{ padding:'7px 12px', textAlign:'left', fontSize:'0.76rem', fontWeight:700, color:'#475569' }}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {comparison.per_model?.map(m => (
                      <tr key={m.model} style={{ borderBottom:'1px solid #f1f5f9' }}>
                        <td style={{ padding:'7px 12px', fontSize:'0.85rem', fontWeight:600 }}>{m.model_name}</td>
                        <td style={{ padding:'7px 12px', fontSize:'0.82rem' }}>{(m.accuracy_a*100).toFixed(1)}%</td>
                        <td style={{ padding:'7px 12px', fontSize:'0.82rem' }}>{(m.accuracy_b*100).toFixed(1)}%</td>
                        <td style={{ padding:'7px 12px', fontSize:'0.82rem', fontWeight:700, color: m.delta>0?'#15803d':m.delta<0?'#b91c1c':'#94a3b8' }}>
                          {m.delta>0?'+':''}{(m.delta*100).toFixed(2)}%
                        </td>
                        <td style={{ padding:'7px 12px' }}>{m.improved?'✅':'↓'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {comparison.recommendation === 'promote' && (
                <button style={{ ...btnP, marginTop:14 }} onClick={()=>promote(cmpB)} disabled={promoting}>
                  {promoting?'Promoting…':'⬆ Promote Version B to Production'}
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
