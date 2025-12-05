import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import api from '../services/api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import ReactMarkdown from 'react-markdown'

export default function DoctorDashboard() {
	const { user } = useAuth()
	const [patientsToday, setPatientsToday] = useState([])
	const [selected, setSelected] = useState(null)
	const [history, setHistory] = useState([])
	const [predictions, setPredictions] = useState([])
	const [intervention, setIntervention] = useState(null)
	const [loading, setLoading] = useState(true)
	const [loadingHistory, setLoadingHistory] = useState(false)
	const [loadingPredictions, setLoadingPredictions] = useState(false)
	const [error, setError] = useState('')
	const [predictionsError, setPredictionsError] = useState('')
	const [expandedId, setExpandedId] = useState(null)
	const [submittingNotes, setSubmittingNotes] = useState({})

	// Map score (0-3) to PHQ-9 option labels; used if server doesn't send selected option text
	const scoreToLabel = (score) => {
		switch (Number(score)) {
			case 0: return 'Not at all'
			case 1: return 'Several days'
			case 2: return 'More than half the days'
			case 3: return 'Nearly every day'
			default: return String(score)
		}
	}

	const formatDateOnly = (dateString) => {
		try {
			const d = new Date(dateString)
			if (!isNaN(d.getTime())) return d.toLocaleDateString()
			// Fallback: try to split if server already formatted
			return String(dateString).split(' ')[0]
		} catch {
			return String(dateString)
		}
	}

	useEffect(() => {
		let mounted = true
		async function load() {
			try {
				setLoading(true)
				const today = await api.getTodaysSubmissions()
				if (!mounted) return
				setPatientsToday(today)
			} catch (e) { if (mounted) setError(e.message || 'Failed to load') }
			finally { if (mounted) setLoading(false) }
		}
		load()
		return () => { mounted = false }
	}, [user?.id])

	const loadTodaySubmissions = async () => {
		try {
			setLoading(true)
			const today = await api.getTodaysSubmissions()
			setPatientsToday(today)
		} catch (e) { setError(e.message || 'Failed to load') }
		finally { setLoading(false) }
	}

	const selectPatient = async (p) => {
		setSelected(p)
		setLoadingHistory(true)
		setLoadingPredictions(true)
		setError('')
		setPredictionsError('')
		try {
			// Load both history and predictions in parallel
			const [historyRes, predictionsRes] = await Promise.allSettled([
				api.getPatientConsultations(p.user_id || p.id),
				api.getPredictions(p.user_id || p.id)
			])
			
			// Handle history response
			if (historyRes.status === 'fulfilled') {
				setHistory(historyRes.value)
				setExpandedId(historyRes.value[0]?.id || null)
			} else {
				setError(historyRes.reason.message || 'Failed to load history')
			}
			
			// Handle predictions response
			if (predictionsRes.status === 'fulfilled') {
				const predictionData = predictionsRes.value || {}
				setPredictions(predictionData.predictions || [])
				setIntervention(predictionData.intervention || null)
			} else {
				setPredictionsError(predictionsRes.reason.message || 'No prediction data available')
				setPredictions([])
				setIntervention(null)
			}
		} catch (e) { 
			setError(e.message || 'Failed to load data')
		}
		finally { 
			setLoadingHistory(false)
			setLoadingPredictions(false)
		}
	}

	const handleSubmitNotes = async (recordId, notes) => {
		setSubmittingNotes(prev => ({ ...prev, [recordId]: true }))
		try {
			await api.updateDoctorNotes(recordId, notes)
			
			// Clear the textarea after successful submission
			const textarea = document.getElementById(`notes-${recordId}`)
			if (textarea) {
				textarea.value = ''
			}
			
			// Re-fetch both history and predictions to get updated data
			if (selected) {
				setLoadingHistory(true)
				setLoadingPredictions(true)
				setError('')
				setPredictionsError('')
				
				try {
					const [historyRes, predictionsRes] = await Promise.allSettled([
						api.getPatientConsultations(selected.user_id || selected.id),
						api.getPredictions(selected.user_id || selected.id)
					])
					
					if (historyRes.status === 'fulfilled') {
						setHistory(historyRes.value)
						// Keep the current expanded card expanded after refresh
						const currentExpanded = historyRes.value.find(h => h.id === recordId)
						setExpandedId(currentExpanded ? recordId : (historyRes.value[0]?.id || null))
					} else {
						setError(historyRes.reason.message || 'Failed to refresh history')
					}
					
					if (predictionsRes.status === 'fulfilled') {
						const predictionData = predictionsRes.value || {}
						setPredictions(predictionData.predictions || [])
						setIntervention(predictionData.intervention || null)
						setPredictionsError('')
					} else {
						setPredictionsError(predictionsRes.reason.message || 'No prediction data available')
						setPredictions([])
						setIntervention(null)
					}
					// Also refresh today's submissions to clear any notifications
					await loadTodaySubmissions()
				} catch (e) {
					setError(e.message || 'Failed to refresh data')
				} finally {
					setLoadingHistory(false)
					setLoadingPredictions(false)
				}
			}
		} catch (e) {
			setError(e.message || 'Failed to update notes')
			if ((e.message || '').toLowerCase().includes('already present')) {
				try {
					if (selected) {
						setLoadingHistory(true)
						setLoadingPredictions(true)
						const [historyRes, predictionsRes] = await Promise.allSettled([
							api.getPatientConsultations(selected.user_id || selected.id),
							api.getPredictions(selected.user_id || selected.id)
						])
						if (historyRes.status === 'fulfilled') {
							setHistory(historyRes.value)
							const currentExpanded = historyRes.value.find(h => h.id === recordId)
							setExpandedId(currentExpanded ? recordId : (historyRes.value[0]?.id || null))
						}
						if (predictionsRes.status === 'fulfilled') {
							const predictionData = predictionsRes.value || {}
							setPredictions(predictionData.predictions || [])
							setIntervention(predictionData.intervention || null)
							setPredictionsError('')
						} else {
							setPredictionsError(predictionsRes.reason.message || 'No prediction data available')
							setPredictions([])
							setIntervention(null)
						}
						// Also refresh today's submissions to clear any notifications
						await loadTodaySubmissions()
					}
				} finally {
					setLoadingHistory(false)
					setLoadingPredictions(false)
				}
			}
		} finally {
			setSubmittingNotes(prev => ({ ...prev, [recordId]: false }))
		}
	}

	const toggleExpanded = (recordId) => {
		setExpandedId(expandedId === recordId ? null : recordId)
	}

	const goHome = async () => {
		setSelected(null)
		setHistory([])
		setPredictions([])
		setIntervention(null)
		setExpandedId(null)
		setError('')
		setPredictionsError('')
		await loadTodaySubmissions()
	}

	const refreshHistory = async () => {
		if (!selected) return
		setLoadingHistory(true)
		setLoadingPredictions(true)
		try {
			// Refresh both history and predictions
			const [historyRes, predictionsRes] = await Promise.allSettled([
				api.getPatientConsultations(selected.user_id || selected.id),
				api.getPredictions(selected.user_id || selected.id)
			])
			
			if (historyRes.status === 'fulfilled') {
				setHistory(historyRes.value)
				setExpandedId(historyRes.value[0]?.id || null)
			}
			
			if (predictionsRes.status === 'fulfilled') {
				const predictionData = predictionsRes.value || {}
				setPredictions(predictionData.predictions || [])
				setIntervention(predictionData.intervention || null)
				setPredictionsError('')
			} else {
				setPredictionsError(predictionsRes.reason.message || 'No prediction data available')
				setPredictions([])
				setIntervention(null)
			}
		} catch (e) {
			setError(e.message || 'Failed to refresh data')
		} finally {
			setLoadingHistory(false)
			setLoadingPredictions(false)
		}
	}

	// Build separate datasets for chart lines
	const actualData = predictions.filter(p => !p.is_predicted)
	const predictedData = predictions.filter(p => p.is_predicted)

	return (
		<div className="card" style={{ maxWidth: 1400 }}>
			<h1>Doctor Dashboard</h1>
			
			{/* Breadcrumbs */}
			{selected && (
				<nav className="breadcrumbs" style={{ marginBottom: 16, fontSize: 14 }}>
					<button onClick={goHome} className="breadcrumb-link">Home</button>
					<span style={{ margin: '0 8px' }}>›</span>
					<button onClick={refreshHistory} className="breadcrumb-link" style={{ textDecoration: 'underline' }}>{selected.first_name} {selected.last_name} (History)</button>
				</nav>
			)}

			{error && <div className="error" style={{ marginBottom: 12 }}>{error}</div>}
			
			{!selected ? (
				// Home view - Patient list
				<div>
					<h3>Today's Submissions</h3>
					{loading ? <p>Loading...</p> : (
						<ul className="list">
							{patientsToday.map((p) => (
								<li key={p.user_id} className="list-item">
									<button className="btn btn-secondary" onClick={() => selectPatient(p)}>
										<span>{p.first_name} {p.last_name}</span>
									</button>
									{p.notification ? <span aria-label="notification" title="New" style={{ width: 10, height: 10, background: '#ef4444', borderRadius: 9999 }} /> : null}
								</li>
							))}
						</ul>
					)}
				</div>
			) : (
				// Selected patient view - Intervention + Grid (Chart left, History right)
				<div className="history-layout" style={{ display: 'grid', gridTemplateColumns: '7fr 3fr', gridTemplateRows: 'auto 1fr', gap: 24, alignItems: 'start' }}>
					{/* Intervention Card - top-left */}
					<div className="card" style={{ gridColumn: '1 / 2', gridRow: '1 / 2' }}>
						<h3>Recommended Intervention</h3>
						{loadingPredictions ? (
							<p>Loading intervention...</p>
						) : intervention?.intervention ? (
							<div className="prose" style={{ maxWidth: '100%' }}>
								<ReactMarkdown>{intervention.intervention}</ReactMarkdown>
							</div>
						) : (
							<p style={{ color: '#64748b' }}>No intervention available</p>
						)}
					</div>

					{/* Left - Line Chart (bottom-left) */}
					<div className="card" style={{ gridColumn: '1 / 2', gridRow: '2 / 3' }}>
						<h3>PHQ-9 Score Trend</h3>
						<div style={{ height: 420 }}>
							{loadingPredictions ? (
								<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
									<p>Loading predictions...</p>
								</div>
							) : predictionsError ? (
								<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
									<p>{predictionsError}</p>
								</div>
							) : predictions.length === 0 ? (
								<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
									<p>No prediction data available</p>
								</div>
							) : (
								<ResponsiveContainer width="100%" height="100%">
									<LineChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
										<CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
										<XAxis 
											dataKey="consultation_seq"
											type="number"
											domain={['dataMin', 'dataMax']}
											stroke="#64748b"
											fontSize={12}
											tick={{ fill: '#64748b' }}
										/>
										<YAxis 
											stroke="#64748b"
											fontSize={12}
											tick={{ fill: '#64748b' }}
											domain={[0, 'dataMax + 2']}
										/>
										<Tooltip />
										<Legend />
										{/* Actual (non-predicted) */}
										<Line data={actualData} type="monotone" dataKey="phq9_total_score" name="Actual" stroke="#22c55e" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
										{/* Predicted */}
										<Line data={predictedData} type="monotone" dataKey="phq9_total_score" name="Predicted" stroke="#0ea5e9" strokeWidth={2.5} strokeDasharray="5 5" dot={{ r: 3 }} activeDot={{ r: 5 }} />
									</LineChart>
								</ResponsiveContainer>
							)}
						</div>
					</div>

					{/* Right - Consultation History spans both rows */}
					<div className="card" style={{ gridColumn: '2 / 3', gridRow: '1 / 3', position: 'sticky', top: 20, maxHeight: '85vh', overflow: 'hidden' }}>
						<h3>Consultation History - {selected.first_name} {selected.last_name}</h3>
						{loadingHistory ? (
							<p>Loading history...</p>
						) : history.length === 0 ? (
							<p>No consultations yet.</p>
						) : (
							<div className="form" style={{ gap: 12, height: 'calc(85vh - 80px)', overflowY: 'auto' }}>
								{history.map((c) => (
									<div key={c.id} className="question-block">
										<div className="question-heading" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
											<div>
												<strong>Consultation #{c.consultation_number}</strong>
												<div><small>{formatDateOnly(c.submitted_at)}</small></div>
											</div>
											<button 
												className="icon-btn" 
												onClick={() => toggleExpanded(c.id)}
												aria-label={expandedId === c.id ? 'Collapse' : 'Expand'}
											>
												{expandedId === c.id ? (
													<svg className="chevron" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
														<path d="M6 15l6-6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
													</svg>
												) : (
													<svg className="chevron" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
														<path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
													</svg>
												)}
											</button>
										</div>
										{expandedId === c.id && (
											<div className="form" style={{ marginTop: 8 }}>
												<div className="history-meta" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
													<div className="list-item"><span>Total score</span><strong>{c.total_score ?? '—'}</strong></div>
													<div className="list-item"><span>Severity</span><strong>{c.severity ?? '—'}</strong></div>
													<div className="list-item"><span>Q9 flag</span><strong>{String(c.q9_flag ?? '—')}</strong></div>
													<div className="list-item"><span>MDD assessment</span><strong>{c.mdd_assessment ?? '—'}</strong></div>
												</div>
												<div>
													<label className="label">Responses</label>
													<div className="list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 8 }}>
														{Array.isArray(c.questions) && c.questions.length > 0 ? (
															c.questions.map((q) => (
																<div key={q.question_id || q.id} className="list-item" style={{ gap: 8 }}>
																	<div className="truncate-2" style={{ flex: 1 }}>{q.question || q.question_text}</div>
																	<div style={{ fontWeight: 700 }}>{q.selected_option ?? q.value ?? q.answer ?? '—'}</div>
																</div>
															))
														) : (
															Object.entries(c.responses || {}).map(([qid, val]) => {
																const questionMap = {
																	'1': 'Little interest or pleasure in doing things',
																	'2': 'Feeling down, depressed, or hopeless',
																	'3': 'Trouble falling or staying asleep, or sleeping too much',
																	'4': 'Feeling tired or having little energy',
																	'5': 'Poor appetite or overeating',
																	'6': 'Feeling bad about yourself — or that you are a failure or have let yourself or your family down',
																	'7': 'Trouble concentrating on things, such as reading the newspaper or watching television',
																	'8': 'Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual',
																	'9': 'Thoughts that you would be better off dead, or of hurting yourself in some way'
																};
																const questionText = questionMap[qid] || `Question ${qid}`;
																return (
																	<div key={qid} className="list-item" style={{ gap: 8 }}>
																		<div className="truncate-2" style={{ flex: 1 }}>{questionText}</div>
																		<div style={{ fontWeight: 700 }}>{val}</div>
																	</div>
																);
															})
														)}
													</div>
												</div>
												<div>
													<label className="label">Doctor Notes</label>
													{c.doctor_notes !== null && c.doctor_notes !== undefined ? (
														<div className="list-item" style={{ whiteSpace: 'pre-wrap' }}>
															{c.doctor_notes}
														</div>
													) : (
														<div className="form" style={{ gap: 8 }}>
															<textarea id={`notes-${c.id}`} rows={3} placeholder="Add doctor notes..." />
															<div>
																<button
																	className="btn btn-primary"
																	disabled={Boolean(submittingNotes[c.id])}
																	onClick={() => {
																		const el = document.getElementById(`notes-${c.id}`)
																		const value = el ? el.value.trim() : ''
																		if (!value) return
																		handleSubmitNotes(c.id, value)
																	}}
																>
																	{submittingNotes[c.id] ? 'Saving...' : 'Save Notes'}
																</button>
															</div>
														</div>
													)}
												</div>
											</div>
										)}
									</div>
								))}
							</div>
						)}
					</div>
				</div>
			)}
		</div>
	)
}



