import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'
import api from '../services/api'
import { useAuth } from '../context/AuthContext'

export default function UserPredictions({ userId: userIdProp }) {
	const { user } = useAuth()
	const userId = userIdProp || user?.id
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState('')
	const [predictions, setPredictions] = useState([])
	const [intervention, setIntervention] = useState(null)

	useEffect(() => {
		let mounted = true
		async function load() {
			if (!userId) return
			try {
				setLoading(true)
				setError('')
				const data = await api.getPredictions(userId)
				if (!mounted) return
				// API returns { predictions: [...], intervention: {...} }
				setPredictions(Array.isArray(data?.predictions) ? data.predictions : [])
				setIntervention(data?.intervention || null)
			} catch (e) {
				if (mounted) setError(e?.message || 'Failed to fetch predictions')
			} finally {
				if (mounted) setLoading(false)
			}
		}
		load()
		return () => { mounted = false }
	}, [userId])

	const actualData = useMemo(() => predictions.filter(p => !p.is_predicted), [predictions])
	const predictedData = useMemo(() => predictions.filter(p => p.is_predicted), [predictions])

	return (
		<div className="mt-6">
			{/* Intervention Card */}
			<div className="bg-white shadow-md rounded-2xl p-6 mb-4">
				<h2 className="text-lg font-semibold text-slate-800 mb-3">Recommended Intervention</h2>
				{loading ? (
					<p className="text-slate-500">Loading...</p>
				) : error ? (
					<p className="text-rose-600">{error}</p>
				) : intervention?.intervention ? (
					<div className="prose prose-slate max-w-none">
						<ReactMarkdown>{intervention.intervention}</ReactMarkdown>
					</div>
				) : (
					<p className="text-slate-500">No intervention available.</p>
				)}
			</div>

			{/* Grid: Line Chart (Left) and History (Right) */}
			<div className="grid grid-cols-1 md:[grid-template-columns:70%_30%] gap-4">
				{/* Line Chart Card */}
				<div className="bg-white shadow-md rounded-2xl p-4">
					<h3 className="text-base font-semibold text-slate-800 mb-2">PHQ-9 Score Trend</h3>
					<div className="h-[360px]">
						{loading ? (
							<div className="h-full flex items-center justify-center text-slate-500">Loading chart...</div>
						) : error ? (
							<div className="h-full flex items-center justify-center text-rose-600">{error}</div>
						) : predictions.length === 0 ? (
							<div className="h-full flex items-center justify-center text-slate-500">No data</div>
						) : (
							<ResponsiveContainer width="100%" height="100%">
								<LineChart margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
									<CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
									<XAxis dataKey="consultation_seq" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: '#64748b', fontSize: 12 }} stroke="#64748b" />
									<YAxis tick={{ fill: '#64748b', fontSize: 12 }} stroke="#64748b" domain={[0, 'dataMax + 2']} />
									<Tooltip />
									<Legend />
									<Line data={actualData} type="monotone" dataKey="phq9_total_score" name="Actual" stroke="#22c55e" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
									<Line data={predictedData} type="monotone" dataKey="phq9_total_score" name="Predicted" stroke="#0ea5e9" strokeWidth={2.5} strokeDasharray="5 5" dot={{ r: 3 }} activeDot={{ r: 5 }} />
								</LineChart>
							</ResponsiveContainer>
						)}
					</div>
				</div>

				{/* History Card */}
				<div className="bg-white shadow-md rounded-2xl p-4">
					<h3 className="text-base font-semibold text-slate-800 mb-2">Consultations</h3>
					<div className="max-h-[500px] overflow-y-auto pr-1">
						{loading ? (
							<p className="text-slate-500">Loading...</p>
						) : error ? (
							<p className="text-rose-600">{error}</p>
						) : predictions.length === 0 ? (
							<p className="text-slate-500">No consultations</p>
						) : (
							<div>
								{predictions.map((p) => (
									<div key={p.id} className="bg-white rounded-xl p-3 mb-2 shadow-sm hover:shadow-md transition border border-slate-100">
										<div className="flex items-center justify-between">
											<div className="text-sm text-slate-700 font-medium">Consultation #{p.consultation_seq}</div>
											<div className="text-sm"><span className="inline-flex items-center gap-1">
												<span className={p.is_predicted ? 'w-2 h-2 rounded-full bg-sky-500' : 'w-2 h-2 rounded-full bg-emerald-500'} />
												<span className="text-slate-600">Score:</span>
												<span className="font-semibold text-slate-800">{p.phq9_total_score}</span>
											</span></div>
										</div>
									</div>
								))}
							</div>
						)}
					</div>
				</div>
			</div>
		</div>
	)
}


