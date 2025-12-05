import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://192.168.1.81:5050/api",
  headers: {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
  },
});

function wait(ms = 300) { return new Promise((r) => setTimeout(r, ms)) }

function generateSessionToken(user) {
	return `session.${btoa(`${user.user_id || user.id}:${user.role}:${Date.now()}`)}`
}

const api = {
	async register(payload) {
		try {
			const body = {
				emailid: payload.email || payload.emailid,
				username: payload.username,
				firstname: payload.firstname,
				lastname: payload.lastname,
				age: Number(payload.age),
				gender: payload.gender,
				industry: payload.industry,
				profession: payload.profession,
				password: payload.password,
				role: payload.role,
			}
			const { data, status } = await apiClient.post('/register', body)
			if (status >= 400 || data?.status === 'Failed' || data?.error) {
				throw new Error(data?.message || data?.error || 'Registration failed')
			}
			return { message: data?.message || 'Registered', data: data?.data }
		} catch (err) {
			throw new Error(err?.response?.data?.message || err?.message || 'Registration failed')
		}
	},

	async login({ username, password }) {
		try {
			const { data, status } = await apiClient.post('/login', { username, password })
			if (status >= 400 || data?.status === 'Failed' || data?.error) {
				throw new Error(data?.message || data?.error || 'Invalid username or password')
			}
			const user = { id: String(data.data.user_id), username, role: data.data.role }
			const access_token = generateSessionToken({ user_id: user.id, role: user.role })
			return { access_token, user }
		} catch (err) {
			throw new Error(err?.response?.data?.message || err?.message || 'Invalid username or password')
		}
	},

	async getConsultationStatus(userId) {
		const { data } = await apiClient.get(`/phq9/check-submission`, { params: { user_id: userId } })
		if (data?.status !== 'success') throw new Error(data?.message || 'Failed to fetch status')
		return { submitted_today: Boolean(data?.data?.hasSubmittedToday) }
	},

	async getQuestions() {
		const { data } = await apiClient.get('/phq9/questions')
		if (data?.status !== 'success') throw new Error(data?.message || 'Failed to fetch questions')
		return data.data // [{question_id, question}]
	},

	async submitConsultation({ userId, responses, patientsNotes }) {
		try {
			const body = { user_id: Number(userId), responses, patients_notes: patientsNotes }
			const { data, status } = await apiClient.post('/phq9/submit', body)
			if (status >= 400 || data?.status !== 'success') throw new Error(data?.message || 'Submission failed')
			return data
		} catch (err) {
			throw new Error(err?.response?.data?.message || err?.message || 'Submission failed')
		}
	},

	async getTodaysSubmissions() {
		const { data } = await apiClient.get('/phq9/todays-submissions')
		if (data?.status !== 'success') throw new Error(data?.message || 'Failed to fetch today submissions')
		return data.data
	},

	async getPatientConsultations(userId) {
		const { data } = await apiClient.get('/phq9/history', { params: { user_id: userId } })
		if (data?.status !== 'success') throw new Error(data?.message || 'Failed to fetch history')
		return data.data
	},

	async updateDoctorNotes(recordId, doctorNotes) {
		try {
			const { data, status } = await apiClient.put('/phq9/update-doctor-notes', { id: recordId, doctor_notes: doctorNotes })
			if (status >= 400 || data?.status !== 'success') throw new Error(data?.message || 'Failed to update notes')
			return data
		} catch (err) {
			throw new Error(err?.response?.data?.message || err?.message || 'Failed to update notes')
		}
	},

	async getPredictions(userId) {
		try {
			const { data, status } = await apiClient.get(`/predictions/${userId}`)
			if (status >= 400 || data?.status !== 'success') throw new Error(data?.message || 'Failed to fetch predictions')
			return data.data
		} catch (err) {
			const msg = err?.response?.data?.message || err?.message || ''
			const code = err?.response?.status
			if (code === 404 || /no prediction records found/i.test(msg)) {
				return { predictions: [], intervention: null }
			}
			throw new Error(msg || 'Failed to fetch predictions')
		}
	},
}

export default api
