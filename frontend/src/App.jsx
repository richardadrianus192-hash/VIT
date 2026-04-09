// frontend/src/App.jsx
// VIT Sports Intelligence Network — v3.0.0
// Layout: Fixed sidebar + main content area

import { useEffect, useMemo, useState } from 'react'
import { fetchHealth, fetchHistory, fetchPicks, predictMatch, API_KEY } from './api'
import AdminPanel      from './AdminPanel'
import AccumulatorPanel from './AccumulatorPanel'
import TrainingPanel   from './TrainingPanel'
import AnalyticsPanel  from './AnalyticsPanel'
import OddsPanel       from './OddsPanel'
import MatchDetail     from './MatchDetail'
import './App.css'

const DEFAULT_FORM = {
  home_team: '', away_team: '', league: 'premier_league',
  kickoff_time: new Date().toISOString().slice(0, 16),
  home: 2.0, draw: 3.2, away: 3.8,
}

const NAV = [
  { group: 'Predict',
    items: [
      { id: 'dashboard',   icon: '📊', label: 'Dashboard' },
      { id: 'picks',       icon: '🏅', label: 'Picks' },
      { id: 'accumulator', icon: '🎰', label: 'Accumulators' },
    ]
  },
  { group: 'Market',
    items: [
      { id: 'odds',        icon: '💎', label: 'Odds & Arbitrage' },
      { id: 'analytics',   icon: '📈', label: 'Analytics' },
    ]
  },
  { group: 'System',
    items: [
      { id: 'training',    icon: '🧠', label: 'Training' },
      { id: 'admin',       icon: '⚙️',  label: 'Admin' },
    ]
  },
]

function PickCard({ pick, onOpen }) {
  const edge = ((pick.edge || 0) * 100).toFixed(2)
  const isCertified = pick.pick_type === 'certified'
  return (
    <div className={`pick-card ${isCertified ? 'certified' : 'high-conf'}`} onClick={() => onOpen(pick.match_id)}>
      <div className="pick-card-badge">{isCertified ? '🏅 Certified' : '⚡ High Confidence'}</div>
      <div className="pick-card-teams">{pick.home_team} <span>vs</span> {pick.away_team}</div>
      <div className="pick-card-stats">
        <span>🎯 <strong>{pick.bet_side?.toUpperCase()}</strong></span>
        <span style={{ color: '#10b981' }}>📈 +{edge}% edge</span>
        <span>🎲 {pick.entry_odds?.toFixed(2)} odds</span>
        <span>💰 {((pick.recommended_stake || 0) * 100).toFixed(2)}% stake</span>
      </div>
      <div className="pick-card-models">
        <span>🤖 {pick.num_models} models</span>
        <span>✅ {pick.model_agreement_pct}% agree</span>
        <span>🧠 Conf: {((pick.avg_1x2_confidence || 0) * 100).toFixed(0)}%</span>
      </div>
      <div className="pick-card-footer">
        {new Date(pick.timestamp).toLocaleString('en-GB', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' })}
        <span className="pick-view-link">View →</span>
      </div>
    </div>
  )
}

function App() {
  const [activeTab, setActiveTab]   = useState('dashboard')
  const [health, setHealth]         = useState(null)
  const [history, setHistory]       = useState([])
  const [picks, setPicks]           = useState(null)
  const [form, setForm]             = useState(DEFAULT_FORM)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading]       = useState(false)
  const [picksLoading, setPL]       = useState(false)
  const [error, setError]           = useState('')
  const [page, setPage]             = useState(0)
  const [selectedMatchId, setSelectedMatchId] = useState(null)
  const itemsPerPage = 10

  const marketOdds = useMemo(
    () => ({ home: parseFloat(form.home), draw: parseFloat(form.draw), away: parseFloat(form.away) }),
    [form.home, form.draw, form.away],
  )

  useEffect(() => {
    fetchHealthStatus()
    loadHistory()
    const id = setInterval(fetchHealthStatus, 15000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => { if (activeTab === 'picks' && !picks) loadPicks() }, [activeTab])

  async function fetchHealthStatus() {
    try { setHealth(await fetchHealth()) } catch (e) { /* silent */ }
  }

  async function loadHistory() {
    try { const r = await fetchHistory(100, 0); setHistory(r.predictions || []); setPage(0) }
    catch (e) { /* silent */ }
  }

  async function loadPicks() {
    setPL(true)
    try { setPicks(await fetchPicks()) } catch (e) { setError(e.message) } finally { setPL(false) }
  }

  async function submitPrediction(e) {
    e.preventDefault()
    if (!form.home_team.trim() || !form.away_team.trim()) { setError('Please enter both team names'); return }
    if (form.home_team.trim() === form.away_team.trim()) { setError('Teams must be different'); return }
    setLoading(true); setError(''); setPrediction(null)
    try {
      const res = await predictMatch({
        home_team: form.home_team.trim(), away_team: form.away_team.trim(),
        league: form.league, kickoff_time: new Date(form.kickoff_time).toISOString(),
        market_odds: marketOdds,
      })
      setPrediction(res); await loadHistory(); if (picks) setPicks(null)
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  function updateField(k, v) { setForm(f => ({ ...f, [k]: v })) }

  const paginated = history.slice(page * itemsPerPage, (page + 1) * itemsPerPage)
  const maxPages  = Math.ceil(history.length / itemsPerPage)

  // Status helpers
  const isOnline   = health?.status === 'ok'
  const modelsOk   = (health?.models_loaded || 0) >= 10
  const dbOk       = health?.db_connected

  // Page titles per tab
  const PAGE_META = {
    dashboard:   { title: 'Dashboard',         sub: 'Make a prediction and review recent history' },
    picks:       { title: 'Market Picks',       sub: 'Certified and high-confidence betting recommendations' },
    accumulator: { title: 'Accumulators',       sub: 'Build and evaluate multi-leg accumulators' },
    analytics:   { title: 'Analytics',          sub: 'Accuracy, ROI, CLV tracking and model contribution' },
    odds:        { title: 'Odds & Arbitrage',   sub: 'Multi-bookmaker comparison, arbitrage scanner, injury notes' },
    training:    { title: 'Training Pipeline',  sub: 'Retrain models, compare versions, upload weights' },
    admin:       { title: 'Admin',              sub: 'Data sources, model management, bulk operations' },
  }

  const meta = PAGE_META[activeTab] || {}

  return (
    <div className="app-shell">
      {selectedMatchId && (
        <MatchDetail matchId={selectedMatchId} onClose={() => setSelectedMatchId(null)} />
      )}

      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside className="sidebar">
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="logo-title">⚽ VIT Predict</div>
          <div className="logo-sub">12-Model Ensemble · v3.0.0</div>
        </div>

        {/* Live status */}
        <div className="sidebar-status">
          <div className="status-row">
            <div className={`status-dot ${isOnline ? 'green' : 'red'}`} />
            <span>API <strong>{isOnline ? 'Online' : 'Offline'}</strong></span>
          </div>
          <div className="status-row">
            <div className={`status-dot ${modelsOk ? 'green' : 'yellow'}`} />
            <span>Models <strong>{health?.models_loaded ?? '…'}/12</strong></span>
          </div>
          <div className="status-row">
            <div className={`status-dot ${dbOk ? 'green' : 'red'}`} />
            <span>Database <strong>{dbOk ? 'Connected' : 'Disconnected'}</strong></span>
          </div>
        </div>

        {/* Nav groups */}
        <nav className="sidebar-nav">
          {NAV.map(group => (
            <div key={group.group}>
              <div className="nav-section-label">{group.group}</div>
              {group.items.map(item => (
                <button
                  key={item.id}
                  className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(item.id)}
                >
                  <span className="nav-icon">{item.icon}</span>
                  {item.label}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          VIT Sports Intelligence
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────────── */}
      <main className="main-content">
        {/* Page title */}
        <div className="page-header fade-in" key={activeTab}>
          <h1>{meta.title}</h1>
          {meta.sub && <p>{meta.sub}</p>}
        </div>

        {/* ── Dashboard ── */}
        {activeTab === 'dashboard' && (
          <div className="fade-in">
            {/* Stats bar */}
            <div className="stats-bar">
              <div className="stat-card">
                <div className="stat-label">Total Predictions</div>
                <div className="stat-value blue">{history.length}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Active Models</div>
                <div className={`stat-value ${modelsOk ? 'green' : ''}`}>{health?.models_loaded ?? '—'}/12</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">With Edge (&gt;2%)</div>
                <div className="stat-value purple">
                  {history.filter(h => (h.final_ev || h.edge || 0) > 0.02).length}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Avg Edge</div>
                <div className="stat-value">
                  {history.length
                    ? `${(history.reduce((s, h) => s + (h.final_ev || h.edge || 0), 0) / history.length * 100).toFixed(2)}%`
                    : '—'
                  }
                </div>
              </div>
            </div>

            {/* Prediction form */}
            <div className="panel">
              <div className="panel-header">
                <h2>🎯 Make a Prediction</h2>
              </div>
              <form className="prediction-form" onSubmit={submitPrediction}>
                <div className="form-row">
                  <div className="field-group">
                    <label htmlFor="home_team">Home Team</label>
                    <input id="home_team" type="text" placeholder="e.g., Arsenal"
                      value={form.home_team} onChange={e => updateField('home_team', e.target.value)} required />
                  </div>
                  <div className="field-group">
                    <label htmlFor="away_team">Away Team</label>
                    <input id="away_team" type="text" placeholder="e.g., Chelsea"
                      value={form.away_team} onChange={e => updateField('away_team', e.target.value)} required />
                  </div>
                  <div className="field-group">
                    <label htmlFor="league">League</label>
                    <select id="league" value={form.league} onChange={e => updateField('league', e.target.value)}>
                      <option value="premier_league">Premier League</option>
                      <option value="la_liga">La Liga</option>
                      <option value="bundesliga">Bundesliga</option>
                      <option value="serie_a">Serie A</option>
                      <option value="ligue_1">Ligue 1</option>
                    </select>
                  </div>
                  <div className="field-group">
                    <label htmlFor="kickoff_time">Kickoff Time</label>
                    <input id="kickoff_time" type="datetime-local" value={form.kickoff_time}
                      onChange={e => updateField('kickoff_time', e.target.value)} required />
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.3px', color: 'var(--text-muted)', marginBottom: 10, display: 'block' }}>
                    Market Odds
                  </label>
                  <div className="market-grid">
                    {['home', 'draw', 'away'].map(k => (
                      <div key={k} className="field-group">
                        <label htmlFor={k}>{k.charAt(0).toUpperCase() + k.slice(1)}</label>
                        <input id={k} type="number" min="1.01" step="0.01" value={form[k]}
                          onChange={e => updateField(k, e.target.value)} required />
                      </div>
                    ))}
                  </div>
                </div>

                <button type="submit" className="primary-button" disabled={loading}>
                  {loading ? 'Generating…' : '🔮 Get Prediction'}
                </button>
              </form>

              {error && <div className="alert error" style={{ marginTop: 16 }}>{error}</div>}

              {prediction && (
                <div className="result-card fade-in" style={{ marginTop: 20 }}>
                  <h3>📊 Prediction Results</h3>
                  <dl>
                    {[
                      ['Home Win',   `${(prediction.home_prob * 100).toFixed(1)}%`,          null],
                      ['Draw',       `${(prediction.draw_prob * 100).toFixed(1)}%`,           null],
                      ['Away Win',   `${(prediction.away_prob * 100).toFixed(1)}%`,           null],
                      ['Over 2.5',   prediction.over_25_prob != null ? `${(prediction.over_25_prob * 100).toFixed(1)}%` : '—', null],
                      ['BTTS',       prediction.btts_prob != null ? `${(prediction.btts_prob * 100).toFixed(1)}%` : '—', null],
                      ['Edge',       `${(prediction.final_ev * 100).toFixed(2)}%`,            prediction.final_ev > 0 ? 'var(--success)' : 'var(--danger)'],
                      ['Stake',      `${(prediction.recommended_stake * 100).toFixed(2)}%`,   'var(--primary)'],
                      ['Confidence', `${(prediction.confidence * 100).toFixed(0)}%`,          '#f97316'],
                    ].map(([label, val, color]) => (
                      <div key={label}>
                        <dt>{label}</dt>
                        <dd style={color ? { color, fontWeight: 700 } : {}}>{val}</dd>
                      </div>
                    ))}
                  </dl>
                  <button className="secondary-button" style={{ marginTop: 14 }}
                    onClick={() => setSelectedMatchId(prediction.match_id)}>
                    View Full Detail →
                  </button>
                </div>
              )}
            </div>

            {/* History */}
            {history.length > 0 && (
              <div className="panel">
                <div className="panel-header">
                  <h2>📈 Prediction History</h2>
                  <button className="secondary-button" onClick={loadHistory}>Refresh</button>
                </div>
                <div className="history-table-wrapper">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Match</th>
                        <th>League</th>
                        <th>Home %</th>
                        <th>Draw %</th>
                        <th>Away %</th>
                        <th>Edge</th>
                        <th>Stake</th>
                        <th>Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginated.map(item => (
                        <tr key={`${item.match_id}-${item.timestamp}`} className="history-row-clickable"
                          onClick={() => setSelectedMatchId(item.match_id)}>
                          <td style={{ fontWeight: 600 }}>
                            {item.home_team} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>v</span> {item.away_team}
                          </td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                            {item.league?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || '—'}
                          </td>
                          <td>{(item.home_prob * 100).toFixed(1)}%</td>
                          <td>{(item.draw_prob * 100).toFixed(1)}%</td>
                          <td>{(item.away_prob * 100).toFixed(1)}%</td>
                          <td style={{ color: (item.final_ev || item.edge || 0) > 0 ? 'var(--success)' : 'var(--danger)', fontWeight: 700 }}>
                            {((item.final_ev || item.edge || 0) * 100).toFixed(2)}%
                          </td>
                          <td>{(item.recommended_stake * 100).toFixed(2)}%</td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                            {item.timestamp ? new Date(item.timestamp).toLocaleString('en-US', { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {maxPages > 1 && (
                  <div className="pagination">
                    <button className="secondary-button" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>← Prev</button>
                    <span>Page {page + 1} of {maxPages}</span>
                    <button className="secondary-button" onClick={() => setPage(p => Math.min(maxPages - 1, p + 1))} disabled={page >= maxPages - 1}>Next →</button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Picks ── */}
        {activeTab === 'picks' && (
          <div className="fade-in">
            <div className="panel-header" style={{ marginBottom: 0 }}>
              <div />
              <button className="secondary-button" onClick={loadPicks} disabled={picksLoading}>
                {picksLoading ? 'Loading…' : '↺ Refresh'}
              </button>
            </div>

            {picksLoading && <div className="picks-loading">Loading picks…</div>}

            {picks && !picksLoading && (
              <>
                {picks.certified_picks?.length > 0 && (
                  <div className="picks-section">
                    <h3 className="picks-section-title">🏅 Certified Picks ({picks.certified_count})</h3>
                    <div className="picks-grid">
                      {picks.certified_picks.map(p => <PickCard key={p.match_id} pick={p} onOpen={setSelectedMatchId} />)}
                    </div>
                  </div>
                )}
                {picks.high_confidence_picks?.length > 0 && (
                  <div className="picks-section">
                    <h3 className="picks-section-title">⚡ High Confidence ({picks.high_confidence_count})</h3>
                    <div className="picks-grid">
                      {picks.high_confidence_picks.map(p => <PickCard key={p.match_id} pick={p} onOpen={setSelectedMatchId} />)}
                    </div>
                  </div>
                )}
                {!picks.certified_picks?.length && !picks.high_confidence_picks?.length && (
                  <div className="picks-empty"><div>📊</div><p>No qualifying picks yet. Run some predictions first.</p></div>
                )}
              </>
            )}

            {!picks && !picksLoading && (
              <div className="picks-empty"><div>🏅</div><p>Click Refresh to load picks.</p></div>
            )}
          </div>
        )}

        {/* ── Accumulators ── */}
        {activeTab === 'accumulator' && (
          <div className="fade-in">
            <AccumulatorPanel apiKey={API_KEY} />
          </div>
        )}

        {/* ── Analytics ── */}
        {activeTab === 'analytics' && (
          <div className="fade-in">
            <AnalyticsPanel apiKey={API_KEY} />
          </div>
        )}

        {/* ── Odds & Arbitrage ── */}
        {activeTab === 'odds' && (
          <div className="fade-in">
            <OddsPanel apiKey={API_KEY} />
          </div>
        )}

        {/* ── Training ── */}
        {activeTab === 'training' && (
          <div className="fade-in">
            <TrainingPanel apiKey={API_KEY} />
          </div>
        )}

        {/* ── Admin ── */}
        {activeTab === 'admin' && (
          <div className="fade-in">
            <AdminPanel apiKey={API_KEY} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
