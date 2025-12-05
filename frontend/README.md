# PHQ-9 Medical Consultation Frontend

A React-based frontend application for medical consultation management with PHQ-9 depression screening and AI-powered predictions.

## 🚀 Tech Stack

- **Framework**: React 18 with JSX
- **Build Tool**: Vite
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Routing**: React Router v6
- **State Management**: React Context API
- **Markdown Rendering**: react-markdown

## 📁 Project Structure

```
src/
├── components/
│   ├── ConsultationForm.jsx      # PHQ-9 questionnaire form
│   ├── PatientList.jsx          # Patient listing component
│   ├── PrivateRoute.jsx         # Route protection wrapper
│   └── UserPredictions.jsx      # Predictions & interventions display
├── context/
│   └── AuthContext.jsx          # Authentication state management
├── pages/
│   ├── DoctorDashboard.jsx      # Doctor's main interface
│   ├── LoginPage.jsx           # User login form
│   ├── PatientDashboard.jsx    # Patient's main interface
│   └── RegisterPage.jsx        # User registration form
├── services/
│   └── api.js                  # API client configuration
├── App.jsx                     # Main application component
├── index.css                   # Global styles
└── main.jsx                    # Application entry point
```

## 🛣️ Frontend Routes

### Public Routes
- **`/`** - Redirects to login page
- **`/login`** - Patient login form
- **`/register`** - User registration form

### Protected Routes
- **`/dashboard`** - Main dashboard (role-based)
  - **Patient View**: PHQ-9 consultation form and submission status
  - **Doctor View**: Patient list, consultation history, predictions, and interventions

## 🔐 Authentication

### User Roles
- **Patient**: Can submit PHQ-9 consultations and view their status
- **Doctor**: Can view patient submissions, consultation history, predictions, and add notes

### Login Credentials
- **Doctor Account**: 
  - Username: `rajeev`
  - Password: `rajeev`
- **Patient Accounts**: Register through the registration form

### Quick Doctor Login
- Click "Doctor Login" button in the navigation bar to instantly switch to doctor account
- Automatically logs out current user and logs in as doctor

## 📊 Features

### Patient Dashboard
- **Daily Consultation Form**: PHQ-9 questionnaire with 9 questions
- **Submission Status**: Shows if consultation has been submitted today
- **Patient Notes**: Optional notes field for additional context

### Doctor Dashboard
- **Patient List**: Today's submissions with notification indicators
- **Patient Selection**: Click any patient to view their detailed information
- **Consultation History**: Expandable cards showing:
  - PHQ-9 responses and scores
  - Patient notes
  - Doctor notes (editable)
  - Severity assessment and MDD flags
- **AI Predictions & Interventions**:
  - **Intervention Card**: AI-generated recommendations (markdown formatted)
  - **Line Chart**: PHQ-9 score trends showing:
    - **Green solid line**: Actual scores
    - **Blue dashed line**: Predicted scores
  - **Consultation History Panel**: Scrollable list of all consultations

## 🔌 API Endpoints

### Base Configuration
- **Base URL**: `https://52370fcee23f.ngrok-free.app/api`
- **Headers**: 
  - `Content-Type: application/json`
  - `ngrok-skip-browser-warning: true`

### Authentication APIs
```javascript
POST /api/register
Body: {
  emailid: string,
  username: string,
  firstname: string,
  lastname: string,
  age: number,
  gender: string,
  industry: string,
  profession: string,
  password: string,
  role: string
}

POST /api/login
Body: {
  username: string,
  password: string
}
```

### PHQ-9 APIs
```javascript
GET /api/phq9/check-submission?user_id={userId}
// Returns: { submitted_today: boolean }

GET /api/phq9/questions
// Returns: Array of question objects

POST /api/phq9/submit
Body: {
  user_id: number,
  responses: object,
  patients_notes: string
}

GET /api/phq9/todays-submissions
// Returns: Array of today's patient submissions

GET /api/phq9/history?user_id={userId}
// Returns: Array of patient consultation history

PUT /api/phq9/update-doctor-notes
Body: {
  id: number,
  doctor_notes: string
}
```

### Predictions API
```javascript
GET /api/predictions/{userId}
// Returns: {
//   data: {
//     predictions: Array<{
//       consultation_seq: number,
//       phq9_total_score: number,
//       is_predicted: boolean,
//       dsm5_mdd_assessment_enc: number,
//       relapse: number
//     }>,
//     intervention: {
//       intervention: string, // Markdown formatted
//       created_at: string,
//       id: number,
//       user_id: number
//     }
//   }
// }
```

## 🎨 UI Components

### Navigation Bar
- **Brand**: "Medical Consultation" link
- **Actions**:
  - **Logged Out**: "Patient Login", "Doctor Login", "Register"
  - **Logged In**: User badge, "Login as Doctor", "Logout"

### Dashboard Layouts

#### Patient Dashboard
- Simple card layout with consultation form
- Status messages for submission confirmation

#### Doctor Dashboard
- **Home View**: Patient list with notification badges
- **Patient View**: 
  - **Top Row**: Intervention card (left) + Consultation history (right)
  - **Bottom Row**: Line chart (left) + Consultation history continues (right)
  - **Breadcrumb Navigation**: Home ↔ Patient History

### Responsive Design
- **Desktop**: Two-column layout (70% chart, 30% history)
- **Mobile**: Stacked layout (100% width each)
- **Grid System**: CSS Grid with Tailwind classes

## 🚀 Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables
Create `.env` file (optional):
```env
VITE_API_BASE_URL=http://192.168.1.81:6000
```

## 🔧 Development

### Key Files
- **`src/services/api.js`**: API client configuration and endpoints
- **`src/context/AuthContext.jsx`**: Authentication state management
- **`src/pages/DoctorDashboard.jsx`**: Main doctor interface with predictions
- **`src/components/UserPredictions.jsx`**: Reusable predictions component

### State Management
- **Authentication**: React Context API with localStorage persistence
- **User Data**: Stored in context and localStorage
- **API Calls**: Centralized in `services/api.js`

### Styling
- **Framework**: Tailwind CSS 4
- **Custom Classes**: Defined in `src/index.css`
- **Responsive**: Mobile-first approach with breakpoints

## 📱 Usage Guide

### For Patients
1. **Register**: Create account with personal details
2. **Login**: Use credentials to access dashboard
3. **Daily Consultation**: Complete PHQ-9 questionnaire
4. **Submit**: Review and submit responses with optional notes

### For Doctors
1. **Quick Login**: Click "Doctor Login" button
2. **View Patients**: See today's submissions list
3. **Select Patient**: Click patient name to view details
4. **Review Data**: 
   - Read AI-generated interventions
   - Analyze PHQ-9 score trends
   - Review consultation history
5. **Add Notes**: Edit doctor notes for each consultation
6. **Navigate**: Use breadcrumbs to return to patient list

## 🔍 Troubleshooting

### Common Issues
- **API Connection**: Ensure backend is running and accessible
- **Authentication**: Clear localStorage if login issues persist
- **Charts Not Loading**: Check if Recharts is properly installed
- **Markdown Not Rendering**: Verify react-markdown dependency

### Development Tips
- Use browser dev tools to inspect API calls
- Check console for JavaScript errors
- Verify network requests in Network tab
- Test responsive design on different screen sizes

## 📄 License

This project is part of a medical consultation system. Please ensure compliance with healthcare data regulations when deploying.

---

**Note**: This frontend connects to a backend API. Ensure the backend is running and accessible at the configured URL for full functionality.