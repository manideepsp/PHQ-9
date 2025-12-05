import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import api from '../services/api'
import ConsultationForm from '../components/ConsultationForm'

export default function PatientDashboard() {
	const { user } = useAuth()
	const [loading, setLoading] = useState(true)
	const [status, setStatus] = useState(null)
	const [error, setError] = useState('')
	const [submitting, setSubmitting] = useState(false)
	const [questions, setQuestions] = useState([])

	useEffect(() => {
		let mounted = true
		async function init() {
			try {
				setLoading(true)
				const s = await api.getConsultationStatus(user.id)
				if (!mounted) return
				setStatus(s)
				if (!s.submitted_today) {
					const qs = await api.getQuestions()
					if (!mounted) return
					setQuestions(qs)
				}
			} catch (e) { if (mounted) setError(e.message || 'Failed to load') }
			finally { if (mounted) setLoading(false) }
		}
		init()
		return () => { mounted = false }
	}, [user?.id])

	const handleSubmit = async (answers) => {
		setSubmitting(true)
		setError('')
		try {
			// answers: { q1: '0', ... } convert to { '1': 0, ... }
			const responses = Object.fromEntries(
				Object.entries(answers).filter(([k]) => k !== 'patients_notes').map(([k, v]) => [String(parseInt(k.replace('q', ''), 10)), Number(v)])
			)
			const patientsNotes = answers.patients_notes || ''
			await api.submitConsultation({ userId: user.id, responses, patientsNotes })
			const s = await api.getConsultationStatus(user.id)
			setStatus(s)
		} catch (e) { setError(e.message || 'Submission failed') }
		finally { setSubmitting(false) }
	}

	return (
		<div className="card">
			<h1>Patient Dashboard</h1>
			{loading ? (
				<p>Loading...</p>
			) : error ? (
				<div className="error">{error}</div>
			) : status?.submitted_today ? (
				<div className="success">You have submitted today’s consultation.</div>
			) : (
				<>
					<p>Please complete your daily consultation:</p>
					<ConsultationForm onSubmit={handleSubmit} submitting={submitting} questions={questions} requireNotes />
				</>
			)}
		</div>
	)
}



