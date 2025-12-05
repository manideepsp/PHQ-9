from flask import Blueprint, request, jsonify
from services.user_service import UserService
import pdb

phq9_bp = Blueprint('phq9', __name__)


@phq9_bp.route('/phq9/check-submission', methods=['GET'])
def check_submission():
    """
    Check if a user has submitted the PHQ-9 assessment today.

    Query params:
        user_id: integer
    """
    try:
        user_id_param = request.args.get('user_id')

        # Validate presence
        if user_id_param is None:
            return jsonify({
                "status": "Failed",
                "message": "Missing required query parameter: user_id",
                "data": None
            }), 400

        # Validate type (must be integer)
        try:
            user_id = int(user_id_param)
        except ValueError:
            return jsonify({
                "status": "Failed",
                "message": "Invalid user_id. Must be an integer.",
                "data": None
            }), 400

        result = UserService.check_user_submission_today(user_id)

        # Determine appropriate HTTP status code
        if result.get('status') == 'success':
            return jsonify(result), 200
        elif result.get('message') in ("Missing required query parameter: user_id", "Invalid user_id. Must be an integer."):
            return jsonify(result), 400
        elif result.get('message') == 'User not found':
            return jsonify(result), 404
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            "status": "Failed",
            "message": f"Unhandled error: {str(e)}",
            "data": None
        }), 500


@phq9_bp.route('/phq9/submit', methods=['POST'])
def submit_phq9():
    """
    Submit a PHQ-9 assessment for a patient.

    Body JSON:
    {
      "user_id": 12,
      "responses": {"1": 1, ..., "9": 2},
      "patients_notes": "..."
    }
    """
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({
                "status": "error",
                "message": "Invalid or missing JSON payload.",
                "error": "missing_body"
            }), 400

        result, status_code = UserService.submit_phq9_assessment(payload)
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to submit PHQ-9 assessment.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/questions', methods=['GET'])
def get_phq9_questions():
    """Get the list of PHQ-9 questions."""
    try:
        result, status_code = UserService.get_all_phq9_questions()
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch PHQ-9 questions.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/todays-submissions', methods=['GET'])
def get_todays_submissions():
    """Get today's PHQ-9 submissions with user details and notification flag."""
    try:
        result, status_code = UserService.get_todays_phq9_submissions()
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch today's PHQ-9 submissions.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/history', methods=['GET'])
def get_phq9_history():
    """Get PHQ-9 assessment history for a user."""
    try:
        user_id_param = request.args.get('user_id')

        if user_id_param is None:
            return jsonify({
                "status": "error",
                "message": "Missing required query parameter: user_id",
                "error": "user_id_missing"
            }), 400

        try:
            user_id = int(user_id_param)
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid user_id. Must be an integer.",
                "error": "user_id_invalid"
            }), 400

        result, status_code = UserService.get_phq9_history_by_user(user_id)
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch PHQ-9 assessment history.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/get-questions', methods=['GET'])
def get_phq_9_questions():
    """Get PHQ-9 questions."""
    try:
        result, status_code = UserService.get_phq_9_questions()
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch PHQ-9 questions.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/update-doctor-notes', methods=['PUT'])
def update_doctor_notes():
    """
    Update doctor notes for a PHQ-9 assessment record.
    Only allows update if current doctor_notes is NULL.
    """
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({
                "status": "error",
                "message": "Invalid or missing JSON payload.",
                "data": None
            }), 400

        # Extract required fields
        assessment_id = payload.get('id')
        doctor_notes = payload.get('doctor_notes')

        # Validate both fields are present
        if not assessment_id and not doctor_notes:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: id and doctor_notes",
                "data": None
            }), 400

        result, status_code = UserService.update_doctor_notes(assessment_id, doctor_notes)
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to update doctor notes.",
            "error": str(e)
        }), 500

@phq9_bp.route('/predictions/<int:user_id>', methods=['GET'])
def get_predictions(user_id):
    """Get predictions for a user sorted by consultation_seq."""
    try:
        result, status_code = UserService.get_predictions_by_user(user_id)
        # interventions = UserService.get_interventions_by_user(user_id, result)
        # result['interventions'] = interventions
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch predictions.",
            "error": str(e)
        }), 500

@phq9_bp.route('/phq9/train-model', methods=['GET'])
def train_model():
    """Train the PHQ-9 model."""
    try:
        result, status_code = UserService.train_model()
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to train model.",
            "error": str(e)
        }), 500


