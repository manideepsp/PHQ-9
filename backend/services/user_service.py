from models.user_model import User, db
from utils.security import hash_password, verify_password
from sqlalchemy.exc import IntegrityError
from datetime import date
from models.phq9_assessment_model import Phq9Assessment
from sqlalchemy import func
from models.phq9_question_model import Phq9Question
from models.dsm5_assessment_model import DSM5Assessment
from models.predictions_model import Prediction
from models.interventions_model import Interventions
# Import timeline_prediction lazily inside methods that use it so the app can start
# even when heavy ML libs (tensorflow) are not installed in the container.
# 'interventions' depends on qdrant_client and other optional libs; import lazily only where needed



class UserService:
    """Service class for user-related business logic"""

    @staticmethod
    def get_phq_9_questions():
        """Fetch all PHQ-9 questions."""
        try:
            questions = Phq9Question.query.order_by(Phq9Question.question_id.asc()).all()
            return {
                "status": "success",
                "message": "PHQ-9 questions fetched successfully.",
                "data": [q.to_dict() for q in questions]
            }, 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to fetch PHQ-9 questions.",
                "error": str(e)
            }, 500

    @staticmethod
    def train_model():
        """Train the PHQ-9 model."""
        try:
            # lazy import to avoid requiring tensorflow at module import time
            from services.timeline_prediction import train_lstm_model_for_timeline_prediction

            train_lstm_model_for_timeline_prediction(limit_users=10000)
            return {
                "status": "success",
                "message": "Model trained successfully.",
                "data": None
            },200
        except Exception as e:
            return {
                "status": "error",
                "message": "Failed to train model.",
                "error": str(e)
            },500
    
    @staticmethod
    def register_user(user_data):
        """
        Register a new user in the database
        
        Args:
            user_data (dict): User registration data
            
        Returns:
            dict: Response with status and data
        """
        try:
            # Validate required fields
            required_fields = ['emailid', 'username', 'firstname', 'lastname', 
                             'age', 'gender', 'industry', 'profession', 'password']
            
            for field in required_fields:
                if not user_data.get(field):
                    return {
                        "message": f"Missing required field: {field}",
                        "status": "Failed",
                        "data": None
                    }
            
            # Check if username or email already exists
            existing_user = User.query.filter(
                (User.username == user_data['username']) | 
                (User.emailid == user_data['emailid'])
            ).first()
            
            if existing_user:
                return {
                    "message": "Username or email already exists",
                    "status": "Failed",
                    "data": None
                }
            
            # Hash the password
            password_hash = hash_password(user_data['password'])
            
            # Create new user
            new_user = User(
                emailid=user_data['emailid'],
                username=user_data['username'],
                firstname=user_data['firstname'],
                lastname=user_data['lastname'],
                age=user_data['age'],
                gender=user_data['gender'],
                industry=user_data['industry'],
                profession=user_data['profession'],
                password_hash=password_hash,
                role=user_data.get('role', 'user')
            )
            
            # Add to database
            db.session.add(new_user)
            db.session.commit()
            
            return {
                "message": "User registered successfully",
                "status": "Success",
                "data": {
                    "user_id": new_user.user_id,
                    "username": new_user.username
                }
            }
            
        except IntegrityError:
            db.session.rollback()
            return {
                "message": "Username or email already exists",
                "status": "Failed",
                "data": None
            }
        except Exception as e:
            db.session.rollback()
            import traceback
            
            # Print detailed error information
            print(f"REGISTRATION EXCEPTION OCCURRED:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print(f"Full Traceback:")
            traceback.print_exc()
            print("-" * 50)
            
            return {
                "message": f"Registration failed: {str(e)}",
                "status": "Failed",
                "data": None
            }
    
    @staticmethod
    def login_user(username, password):
        """
        Authenticate user login
        
        Args:
            username (str): Username or email
            password (str): Plain text password
            
        Returns:
            dict: Response with status and data
        """
        try:
            # Find user by username
            user = User.query.filter_by(username=username).first()
            
            if not user:
                return {
                    "message": "Invalid username or password",
                    "status": "Failed",
                    "data": None
                }
            
            # Verify password
            if verify_password(user.password_hash, password):
                return {
                    "message": "Login successful",
                    "status": "Success",
                    "data": {
                        "user_id": user.user_id,
                        "username": user.username,
                        "firstname": user.firstname,
                        "lastname": user.lastname,
                        "age": user.age,
                        "gender": user.gender,
                        "industry": user.industry,
                        "profession": user.profession,
                        "role": user.role
                    }
                }
            else:
                return {
                    "message": "Invalid username or password",
                    "status": "Failed",
                    "data": None
                }
                
        except Exception as e:
            import traceback
            
            # Print detailed error information
            print(f"❌ LOGIN EXCEPTION OCCURRED:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print(f"Full Traceback:")
            traceback.print_exc()
            print("-" * 50)
            
            return {
                "message": f"Login failed: {str(e)}",
                "status": "Failed",
                "data": None
            }

    @staticmethod
    def check_user_submission_today(user_id):
        """
        Check if a user has submitted PHQ-9 assessment today.

        Args:
            user_id (int): The user's ID

        Returns:
            dict: Standardized response indicating submission status
        """
        try:
            # Validate user_id
            if not user_id:
                return {
                    "status": "Failed",
                    "message": "Missing required query parameter: user_id",
                    "data": None
                }

            # Ensure user exists (optional but helpful for clear errors)
            user = User.query.filter_by(user_id=user_id).first()
            if not user:
                return {
                    "status": "Failed",
                    "message": "User not found",
                    "data": None
                }

            # Compare against today's date
            today = date.today()

            exists_today = (
                db.session.query(Phq9Assessment.id)
                .filter(
                    Phq9Assessment.user_id == user_id,
                    func.date(Phq9Assessment.submitted_at) == today  # Compare only the date part

                )
                .first()
            )

            if exists_today:
                return {
                    "status": "success",
                    "message": "User has already submitted the PHQ-9 assessment today.",
                    "data": {"hasSubmittedToday": True}
                }
            else:
                return {
                    "status": "success",
                    "message": "User has not submitted the PHQ-9 assessment today.",
                    "data": {"hasSubmittedToday": False}
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "Failed",
                "message": f"Error checking submission: {str(e)}",
                "data": None
            }

    @staticmethod
    def submit_phq9_assessment(payload):
        """
        Validate and submit a PHQ-9 assessment.

        Expected payload keys:
          - user_id: int
          - responses: dict with keys "1".."9" and values 0..3
          - patients_notes: str

        Returns standardized response.
        """
        from datetime import datetime, date as _date

        try:
            # Basic presence validation
            user_id = payload.get('user_id')
            responses = payload.get('responses')
            patients_notes = payload.get('patients_notes')

            if user_id is None:
                return {
                    "status": "error",
                    "message": "Missing required field: user_id",
                    "error": "user_id is required",
                }, 400

            # Validate user exists
            user = User.query.filter_by(user_id=user_id).first()
            if not user:
                return {
                    "status": "error",
                    "message": "Invalid user_id. User not found.",
                    "error": "user_not_found",
                }, 404

            # Validate responses structure
            if not isinstance(responses, dict):
                return {
                    "status": "error",
                    "message": "Invalid responses. Must be an object/dictionary.",
                    "error": "invalid_responses_type",
                }, 400

            expected_keys = {str(i) for i in range(1, 10)}
            if set(responses.keys()) != expected_keys:
                return {
                    "status": "error",
                    "message": "Invalid responses keys. Must include keys '1'..'9'.",
                    "error": "invalid_responses_keys",
                }, 400

            # Validate each value 0..3
            for k, v in responses.items():
                if not isinstance(v, int) or v < 0 or v > 3:
                    return {
                        "status": "error",
                        "message": f"Invalid response value for key {k}. Must be integer 0..3.",
                        "error": "invalid_response_value",
                    }, 400

            # Validate patients_notes
            if not isinstance(patients_notes, str) or patients_notes.strip() == "":
                return {
                    "status": "error",
                    "message": "Invalid patients_notes. Must be a non-empty string.",
                    "error": "invalid_patients_notes",
                }, 400

            # Calculate total score
            total_score = sum(int(responses[str(i)]) for i in range(1, 10))

            # Build model
            now_ts = datetime.utcnow()
            today_date = _date.today()

            assessment = Phq9Assessment(
                user_id=user_id,
                responses=responses,
                total_score=total_score,
                submitted_at=now_ts,
                doctors_notes=None,
                patients_notes=patients_notes,
                # If the model included assessment_date, we'd set it here. The
                # existing codebase uses submitted_at's date for checks.
            )

            db.session.add(assessment)
            
            # Calculate DSM-5 assessment values
            # Severity calculation
            if total_score >= 20:
                severity = "Severe depression"
            elif total_score >= 15:
                severity = "Moderately severe depression"
            elif total_score >= 10:
                severity = "Moderate depression"
            elif total_score >= 5:
                severity = "Mild depression"
            else:
                severity = "No depression"
            
            # Q9 flag calculation
            q9_flag = responses.get('9', 0) >= 2
            
            # MDD assessment rules
            # Rule 1: Count responses >= 2
            num_responses_ge_2 = sum(1 for v in responses.values() if v >= 2)
            rule1 = num_responses_ge_2 >= 5
            
            # Rule 2: q1 >= 2 or q2 >= 2
            rule2 = responses.get('1', 0) >= 2 or responses.get('2', 0) >= 2
            
            # Final MDD Assessment
            if rule1 and rule2:
                mdd_assessment = "true"
            else:
                mdd_assessment = "false"
            
            # Create DSM-5 assessment record
            dsm_entry = DSM5Assessment(
                user_id=user_id,
                severity=severity,
                q9_flag=q9_flag,
                mdd_assessment=mdd_assessment,
                created_at=now_ts  # same timestamp as PHQ-9 record
            )
            
            db.session.add(dsm_entry)
            db.session.commit()

            return {
                "status": "success",
                "message": "PHQ-9 assessment submitted successfully.",
                "data": {
                    "user_id": user_id,
                    "total_score": total_score,
                    "submitted_at": now_ts.isoformat(),
                    "dsm5_assessment": {
                        "severity": severity,
                        "q9_flag": q9_flag,
                        "mdd_assessment": mdd_assessment
                    }
                }
            }, 201

        except Exception as e:
            db.session.rollback()
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to submit PHQ-9 assessment.",
                "error": str(e)
            }, 500

    @staticmethod
    def get_all_phq9_questions():
        """Fetch all PHQ-9 questions."""
        try:
            questions = Phq9Question.query.order_by(Phq9Question.question_id.asc()).all()
            return {
                "status": "success",
                "message": "PHQ-9 questions fetched successfully.",
                "data": [q.to_dict() for q in questions]
            }, 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to fetch PHQ-9 questions.",
                "error": str(e)
            }, 500

    @staticmethod
    def get_todays_phq9_submissions():
        """
        Return today's PHQ-9 submissions with user details and notification flag.
        Notification is True when doctors_notes is NULL.
        """
        try:
            today = date.today()

            # Join Phq9Assessment with User; restrict to submissions where submitted_at is today
            results = (
                db.session.query(
                    User.user_id,
                    User.firstname.label('first_name'),
                    User.lastname.label('last_name'),
                    Phq9Assessment.doctors_notes,
                )
                .join(User, User.user_id == Phq9Assessment.user_id)
                .filter(func.date(Phq9Assessment.submitted_at) == today)
                .all()
            )

            data = []
            for row in results:
                data.append({
                    "user_id": row.user_id,
                    "first_name": row.first_name,
                    "last_name": row.last_name,
                    "notification": row.doctors_notes is None
                })

            return {
                "status": "success",
                "message": "Today's PHQ-9 submissions fetched successfully.",
                "data": data
            }, 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to fetch today's PHQ-9 submissions.",
                "error": str(e)
            }, 500

    @staticmethod
    def get_phq9_history_by_user(user_id):
        """
        Fetch all PHQ-9 submissions for a user, sorted by submitted_at desc,
        and assign consultation_number in ascending order starting from 1,
        with the latest record having the highest number.
        """
        try:
            # Validate user_id
            if user_id is None:
                return {
                    "status": "error",
                    "message": "Missing required query parameter: user_id",
                    "error": "user_id_missing"
                }, 400

            # Query submissions
            records = (
                Phq9Assessment.query
                .filter(Phq9Assessment.user_id == user_id)
                .order_by(Phq9Assessment.submitted_at.desc())
                .all()
            )

            if not records:
                return {
                    "status": "success",
                    "message": "No PHQ-9 assessments found for this user.",
                    "data": []
                }, 200

            # Fetch related DSM-5 rows by matching id == phq9_assessment.id
            assessment_ids = [r.id for r in records if r.id is not None]
            dsm_by_id = {}
            if assessment_ids:
                dsm_rows = DSM5Assessment.query.filter(DSM5Assessment.id.in_(assessment_ids)).all()
                dsm_by_id = {row.id: row for row in dsm_rows}

            # Build response with consultation_number and merged DSM fields
            total = len(records)
            data = []
            for idx, r in enumerate(records):
                # Latest (first in list) should have highest consultation_number
                consultation_number = total - idx
                submitted_value = r.submitted_at.isoformat(sep=' ', timespec='seconds') if r.submitted_at else None

                # Map DSM extras if present
                dsm = dsm_by_id.get(r.id)
                severity = dsm.severity if dsm else None
                q9_flag = dsm.q9_flag if dsm else None
                mdd_assessment = dsm.mdd_assessment if dsm else None

                data.append({
                    "consultation_number": consultation_number,
                    "id": r.id,
                    "user_id": r.user_id,
                    "doctor_notes": r.doctors_notes,
                    "patient_notes": r.patients_notes,
                    "submitted_at": submitted_value,
                    "total_score": r.total_score,
                    "responses": r.responses,
                    "severity": severity,
                    "q9_flag": q9_flag,
                    "mdd_assessment": mdd_assessment,
                })

            return {
                "status": "success",
                "message": "PHQ-9 assessment history fetched successfully.",
                "data": data
            }, 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to fetch PHQ-9 assessment history.",
                "error": str(e)
            }, 500

    @staticmethod
    def update_doctor_notes(assessment_id, doctor_notes):
        """
        Update doctor notes for a PHQ-9 assessment record.
        Only allows update if current doctor_notes is NULL.
        """
        try:
            # Validate required fields
            if not assessment_id:
                return {
                    "status": "error",
                    "message": "Missing required field: id",
                    "data": None
                }, 400
            
            if not doctor_notes or not isinstance(doctor_notes, str) or doctor_notes.strip() == "":
                return {
                    "status": "error",
                    "message": "Missing required field: doctor_notes",
                    "data": None
                }, 400
            
            # Find the assessment record
            assessment = Phq9Assessment.query.filter_by(id=assessment_id).first()
            
            if not assessment:
                return {
                    "status": "error",
                    "message": "Assessment record not found",
                    "data": None
                }, 404
            
            # Check if doctor_notes already exists
            if assessment.doctors_notes is not None:
                return {
                    "status": "error",
                    "message": "Doctor notes already present and cannot be updated.",
                    "data": None
                }, 400
            
            # Update the doctor notes and commit immediately so this action is durable.
            assessment.doctors_notes = doctor_notes.strip()
            db.session.commit()

            # Try to generate timeline + intervention, but make this non-fatal.
            # If prediction/intervention generation fails for any reason we should not
            # fail the whole endpoint — the notes are already saved and the UI should
            # report the non-blocking warning instead of an error.
            warning = None
            try:
                # lazy import so the app can run without tensorflow present until this feature is invoked
                from services.timeline_prediction import predict_user_timeline

                timeline_dict = predict_user_timeline(str(assessment.user_id), str(assessment.id))

                # lazy import to avoid pulling qdrant_client at startup
                from services.interventions import generate_intervention

                intervention = generate_intervention(timeline_dict, assessment.user_id, assessment.id)
            except Exception as gen_exc:  # non-fatal
                import traceback
                traceback.print_exc()
                warning = f"Intervention generation failed: {str(gen_exc)}"

            response_data = {
                "id": assessment_id,
                "doctor_notes": doctor_notes.strip()
            }

            if warning:
                response_data["warning"] = warning

            return {
                "status": "success",
                "message": "Doctor notes updated successfully.",
                "data": response_data
            }, 200
            
        except Exception as e:
            db.session.rollback()
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to update doctor notes.",
                "error": str(e)
            }, 500

    @staticmethod
    def get_predictions_by_user(user_id):
        """Fetch predictions and latest intervention for a user."""
        try:
            if user_id is None:
                return {
                    "status": "error",
                    "message": "Missing required path parameter: user_id",
                    "error": "user_id_missing"
                }, 400

            # Fetch predictions ordered by consultation_seq ascending
            rows_prediction = (
                Prediction.query
                .filter(Prediction.user_id == user_id)
                .order_by(Prediction.consultation_seq.asc())
                .all()
            )

            # Fetch the latest intervention record
            row_intervention = (
                Interventions.query
                .filter(Interventions.user_id == user_id)
                .order_by(Interventions.created_at.desc())
                .first()
            )

            # Check if predictions exist
            if not rows_prediction:
                return {
                    "status": "error",
                    "message": "No prediction records found for this user.",
                    "data": {
                        "predictions": [],
                        "intervention": None
                    }
                }, 404

            # Prepare response data
            data = {
                "predictions": [row.to_dict() for row in rows_prediction],
                "intervention": row_intervention.to_dict() if row_intervention else None
            }

            return {
                "status": "success",
                "message": "Predictions and intervention fetched successfully.",
                "data": data
            }, 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": "Failed to fetch predictions and intervention.",
                "error": str(e)
            }, 500

        
    # @staticmethod
    # def get_interventions_by_user(user_id, result):
    #     print(result.get('data', []))

    #     try:
    #         if user_id is None:
    #             return {
    #                 "status": "error",
    #                 "message": "Missing required path parameter: user_id",
    #                 "error": "user_id_missing"
    #             }, 400
            
    #         rows = (
    #             Interventions.query
    #             .filter(Interventions.user_id == user_id)
    #             .order_by(Interventions.created_at.desc())
    #             .first()
    #         )

    #         if not rows:
    #             return {
    #                 "status": "error",
    #                 "message": "No intervention records found for this user.",
    #                 "data": []
    #             }, 404

    #         interventions = [rows.to_dict()]
    #         # print("Interventions:", interventions)
    #         return interventions
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         return {
    #             "status": "error",
    #             "message": "Failed to fetch interventions.",
    #             "error": str(e)
    #         }, 500