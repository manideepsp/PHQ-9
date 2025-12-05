import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function RegisterPage() {
	const { register } = useAuth()
	const navigate = useNavigate()

	const [emailid, setEmailid] = useState('')
	const [username, setUsername] = useState('')
	const [firstname, setFirstname] = useState('')
	const [lastname, setLastname] = useState('')
	const [password, setPassword] = useState('')
	const [confirmPassword, setConfirmPassword] = useState('')
	const [age, setAge] = useState('')
	const [gender, setGender] = useState('')
	const [industry, setIndustry] = useState('')
	const [profession, setProfession] = useState('')
	const [role, setRole] = useState('patient')

	const [loading, setLoading] = useState(false)
	const [error, setError] = useState('')
	const [success, setSuccess] = useState('')

	const handleSubmit = async (e) => {
		e.preventDefault()
		setError('')
		setSuccess('')
		if (password !== confirmPassword) {
			setError('Passwords do not match')
			return
		}
		if (!emailid || !username || !firstname || !lastname || !age || !gender || !industry || !profession || !role) {
			setError('Please fill in all required fields')
			return
		}
		setLoading(true)
		const res = await register({
			email: emailid,
			username,
			firstname,
			lastname,
			password,
			age,
			gender,
			industry,
			profession,
			role,
		})
		setLoading(false)
		if (res.success) {
			setSuccess('Registration successful. Redirecting to login...')
			setTimeout(() => navigate('/login'), 900)
		} else {
			setError(res.message || 'Registration failed')
		}
	}

	return (
		<div className="card" style={{ maxWidth: 720 }}>
			<h1>Register</h1>
			<form className="form" onSubmit={handleSubmit}>
				<div className="layout-2col">
					<div>
						<label className="label">Email</label>
						<input className="input" type="email" value={emailid} onChange={(e) => setEmailid(e.target.value)} required />
					</div>
					<div>
						<label className="label">Username</label>
						<input className="input" value={username} onChange={(e) => setUsername(e.target.value)} required />
					</div>
					<div>
						<label className="label">First name</label>
						<input className="input" value={firstname} onChange={(e) => setFirstname(e.target.value)} required />
					</div>
					<div>
						<label className="label">Last name</label>
						<input className="input" value={lastname} onChange={(e) => setLastname(e.target.value)} required />
					</div>
					<div>
						<label className="label">Password</label>
						<input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
					</div>
					<div>
						<label className="label">Confirm password</label>
						<input className="input" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
					</div>
					<div>
						<label className="label">Age</label>
						<input className="input" type="number" min="0" value={age} onChange={(e) => setAge(e.target.value)} required />
					</div>
					<div>
						<label className="label">Gender</label>
						<select className="select" value={gender} onChange={(e) => setGender(e.target.value)} required>
							<option value="" disabled>Select gender</option>
							<option value="Male">Male</option>
							<option value="Female">Female</option>
							<option value="Other">Other</option>
						</select>
					</div>
					<div>
						<label className="label">Industry</label>
						<input className="input" value={industry} onChange={(e) => setIndustry(e.target.value)} required />
					</div>
					<div>
						<label className="label">Profession</label>
						<input className="input" value={profession} onChange={(e) => setProfession(e.target.value)} required />
					</div>
					<div>
						<label className="label">Role</label>
						<select className="select" value={role} onChange={(e) => setRole(e.target.value)} required>
							<option value="patient">Patient</option>
							<option value="doctor">Doctor</option>
						</select>
					</div>
				</div>

				{error && <div className="error">{error}</div>}
				{success && <div className="success">{success}</div>}
				<button className="btn" type="submit" disabled={loading}>{loading ? 'Creating...' : 'Create Account'}</button>
			</form>
			<p style={{ marginTop: 12 }}>Already have an account? <Link to="/login">Login</Link></p>
		</div>
	)
}
