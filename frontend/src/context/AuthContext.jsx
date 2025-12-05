import React, { createContext, useContext, useMemo, useState } from 'react'
import api from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
	const [user, setUser] = useState(() => {
		try {
			const raw = localStorage.getItem('auth_user')
			return raw ? JSON.parse(raw) : null
		} catch { return null }
	})
	const [token, setToken] = useState(() => localStorage.getItem('access_token') || null)
	const [loading, setLoading] = useState(false)

	const login = async (username, password) => {
		setLoading(true)
		try {
			const res = await api.login({ username, password })
			localStorage.setItem('access_token', res.access_token)
			localStorage.setItem('auth_user', JSON.stringify(res.user))
			setToken(res.access_token)
			setUser(res.user)
			return { success: true }
		} catch (err) {
			return { success: false, message: err?.message || 'Login failed' }
		} finally { setLoading(false) }
	}

	const register = async (data) => {
		setLoading(true)
		try {
			await api.register(data)
			return { success: true }
		} catch (err) {
			return { success: false, message: err?.message || 'Registration failed' }
		} finally { setLoading(false) }
	}

	const logout = () => {
		localStorage.removeItem('access_token')
		localStorage.removeItem('auth_user')
		setToken(null)
		setUser(null)
	}

	const value = useMemo(() => ({ user, token, loading, login, logout, register }), [user, token, loading])
	return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
	const ctx = useContext(AuthContext)
	if (!ctx) throw new Error('useAuth must be used within AuthProvider')
	return ctx
}



