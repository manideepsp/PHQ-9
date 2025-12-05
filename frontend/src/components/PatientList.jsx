export default function PatientList({ patients, onSelect }) {
	if (!patients?.length) return <div className="card"><p>No patients found.</p></div>
	return (
		<table className="table">
			<thead>
				<tr>
					<th>Name</th>
					<th>Email</th>
					<th>Action</th>
				</tr>
			</thead>
			<tbody>
				{patients.map((p) => (
					<tr key={p.id}>
						<td>{p.name}</td>
						<td>{p.email}</td>
						<td>
							<button className="btn btn-secondary" onClick={() => onSelect(p)}>View History</button>
						</td>
					</tr>
				))}
			</tbody>
		</table>
	)
}



