import { Routes, Route, Link, useNavigate } from 'react-router-dom'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import PatientDashboard from './pages/PatientDashboard'
import DoctorDashboard from './pages/DoctorDashboard'
import PrivateRoute from './components/PrivateRoute'
import { useAuth } from './context/AuthContext'

export default function App() {
	const { user, logout, login } = useAuth()
	const navigate = useNavigate()

	const handleLogout = () => {
		logout()
		navigate('/login')
	}

	const handleDoctorLogin = async () => {
		// Always reset session to ensure role switch
		logout()
		const res = await login('rajeev', 'rajeev')
		if (res?.success) {
			navigate('/dashboard')
		} else {
			// Fallback to login page on failure
			navigate('/login')
		}
	}

	return (
		<div className="app-container">
			<nav className="top-nav">
				<div className="brand"><Link to="/">Medical Consultation</Link></div>
				<div className="nav-actions">
					{user ? (
						<>
							<span className="user-badge">{user.username || user.email} ({user.role})</span>
							<button className="btn" onClick={handleDoctorLogin}>Login as Doctor</button>
							<button className="btn" onClick={handleLogout}>Logout</button>
						</>
					) : (
						<>
							<Link className="btn-link" to="/login">Patient Login</Link>
							<button className="btn-link" onClick={handleDoctorLogin}>Doctor Login</button>
							<Link className="btn-link" to="/register">Register</Link>
						</>
					)}
				</div>
			</nav>
			<main className="main-content">
				<Routes>
					<Route path="/login" element={<LoginPage />} />
					<Route path="/register" element={<RegisterPage />} />
					<Route
						path="/dashboard"
						element={
							<PrivateRoute>
								{user?.role === 'doctor' ? <DoctorDashboard /> : <PatientDashboard />}
							</PrivateRoute>
						}
					/>
					<Route path="*" element={<LoginPage />} />
				</Routes>
			</main>
		</div>
	)
}



