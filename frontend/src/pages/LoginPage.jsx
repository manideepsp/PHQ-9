import { useState } from 'react'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
	const { login } = useAuth()
	const navigate = useNavigate()
	const location = useLocation()
	const from = location.state?.from?.pathname || '/dashboard'

	const [username, setUsername] = useState('')
	const [password, setPassword] = useState('')
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState('')

	const handleSubmit = async (e) => {
		e.preventDefault()
		setError('')
		setLoading(true)
		const res = await login(username, password)
		setLoading(false)
		if (res.success) {
			navigate(from, { replace: true })
		} else {
			setError(res.message || 'Invalid credentials')
		}
	}

	return (
		<div className="card">
			<h1>Login</h1>
			<form className="form" onSubmit={handleSubmit}>
				<div>
					<label className="label">Username</label>
					<input className="input" value={username} onChange={(e) => setUsername(e.target.value)} required />
				</div>
				<div>
					<label className="label">Password</label>
					<input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
				</div>
				{error && <div className="error">{error}</div>}
				<button className="btn" type="submit" disabled={loading}>{loading ? 'Signing in...' : 'Login'}</button>
			</form>
			<p style={{ marginTop: 12 }}>No account? <Link to="/register">Register</Link></p>
		</div>
	)
}



