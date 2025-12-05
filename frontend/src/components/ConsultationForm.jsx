import { useState, useMemo } from 'react'

const DEFAULT_PHQ9 = [
	{ id: 'q1', text: 'Little interest or pleasure in doing things' },
	{ id: 'q2', text: 'Feeling down, depressed, or hopeless' },
	{ id: 'q3', text: 'Trouble falling or staying asleep, or sleeping too much' },
	{ id: 'q4', text: 'Feeling tired or having little energy' },
	{ id: 'q5', text: 'Poor appetite or overeating' },
	{ id: 'q6', text: 'Feeling bad about yourself — or that you are a failure or have let yourself or your family down' },
	{ id: 'q7', text: 'Trouble concentrating on things, such as reading the newspaper or watching television' },
	{ id: 'q8', text: 'Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual' },
	{ id: 'q9', text: 'Thoughts that you would be better off dead or of hurting yourself in some way' },
]

const OPTIONS = [
	{ label: 'Not at all', value: '0' },
	{ label: 'Several days', value: '1' },
	{ label: 'More than half the days', value: '2' },
	{ label: 'Nearly every day', value: '3' },
]

export default function ConsultationForm({ onSubmit, submitting, questions, requireNotes = false }) {
	const phqQuestions = useMemo(() => {
		if (Array.isArray(questions) && questions.length === 9) {
			return questions.map((q) => ({ id: `q${q.question_id}`, text: q.question }))
		}
		return DEFAULT_PHQ9
	}, [questions])

	const [answers, setAnswers] = useState({})
	const [error, setError] = useState('')

	const handleChange = (qid, value) => setAnswers((prev) => ({ ...prev, [qid]: value }))

	const handleSubmit = (e) => {
		e.preventDefault()
		setError('')
		const allAnswered = phqQuestions.every((q) => answers[q.id] !== undefined)
		if (!allAnswered) {
			setError('Please answer all 9 questions')
			return
		}
		if (requireNotes && !answers.patients_notes) {
			setError('Please add your notes')
			return
		}
		onSubmit(answers)
	}

	return (
		<form className="form" onSubmit={handleSubmit}>
			{phqQuestions.map((q, idx) => (
				<div key={q.id} className="question-block">
					<div className="question-heading">
						<span className="question-number">{idx + 1}.</span>
						<label className="label">{q.text}</label>
					</div>
					<div className="options-row">
						{OPTIONS.map((opt) => (
							<label key={opt.value} className="radio-option">
								<input
									type="radio"
									name={q.id}
									value={opt.value}
									checked={answers[q.id] === opt.value}
									onChange={(e) => handleChange(q.id, e.target.value)}
								/>
								<span>{opt.label}</span>
							</label>
						))}
					</div>
				</div>
			))}
			{requireNotes && (
				<div>
					<label className="label">Patient Notes</label>
					<textarea className="textarea" rows={4} value={answers.patients_notes || ''} onChange={(e) => handleChange('patients_notes', e.target.value)} />
				</div>
			)}
			{error && <div className="error">{error}</div>}
			<button className="btn btn-primary" type="submit" disabled={submitting}>{submitting ? 'Submitting...' : 'Submit Consultation'}</button>
		</form>
	)
}
