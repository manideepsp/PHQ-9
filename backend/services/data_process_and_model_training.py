from models.user_model import User
from models.phq9_assessment_model import Phq9Assessment
from models.dsm5_assessment_model import DSM5Assessment
from models.predictions_model import Prediction
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, distinct
import os
import pickle
from sqlalchemy.orm import sessionmaker
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from config import Config
from models import *
from textblob import TextBlob
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # for saving/loading Python objects
import pdb


# Using database configuration from Config
DATABASE_URL = Config.SQLALCHEMY_DATABASE_URI
 
 
 
DSM5_KEYWORDS = {
    "depressed_mood": [
        "sad", "hopeless", "depressed", "tearful", "empty", "worthless", "down", "gloomy"
    ],
    "anhedonia": [
        "loss of interest", "no pleasure", "lack of motivation", "disinterest", "apathy", "not enjoying"
    ],
    "appetite_change": [
        "loss of appetite", "overeating", "weight loss", "weight gain", "poor appetite", "eating more", "eating less"
    ],
    "sleep_disturbance": [
        "insomnia", "poor sleep", "trouble sleeping", "oversleeping", "restless", "waking early"
    ],
    "psychomotor_change": [
        "restless", "slowed", "agitated", "fidgety", "sluggish"
    ],
    "fatigue": [
        "tired", "fatigue", "no energy", "drained", "lethargic", "exhausted"
    ],
    "worthlessness_guilt": [
        "guilty", "worthless", "self blame", "shame", "regret", "failure"
    ],
    "concentration_difficulty": [
        "can't focus", "forgetful", "indecisive", "trouble concentrating", "mental fog", "confused"
    ],
    "suicidal_thoughts": [
        "suicidal", "wants to die", "hopeless", "no reason to live", "self harm", "ending life"
    ],
}
 
 
def db_engine():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()
 
def extract_data(session, limit_users):

    # 2. Subquery to select top 1000 users
    subq = (
        session.query(distinct(User.user_id).label("user_id"))
        .limit(limit_users)
        .subquery()
    )
 
    results = (
        session.query(
            User.username,
            User.age,
            User.gender,
            User.industry,
            User.profession,
            Phq9Assessment.responses,
            Phq9Assessment.total_score,
            Phq9Assessment.doctors_notes,
            Phq9Assessment.patients_notes,
            DSM5Assessment.severity,
            DSM5Assessment.q9_flag,
            DSM5Assessment.mdd_assessment,
            DSM5Assessment.created_at
        )
        .join(subq, User.user_id == subq.c.user_id)
        .join(Phq9Assessment, User.user_id == Phq9Assessment.user_id)
        .join(DSM5Assessment, Phq9Assessment.id == DSM5Assessment.id)
        .order_by(Phq9Assessment.submitted_at)
        .all()
    )
 
    return results  
 
def extract_user_data(session, user_id):

    results = (
        session.query(
            User.username,
            User.age,
            User.gender,
            User.industry,
            User.profession,
            Phq9Assessment.responses,
            Phq9Assessment.total_score,
            Phq9Assessment.doctors_notes,
            Phq9Assessment.patients_notes,
            DSM5Assessment.severity,
            DSM5Assessment.q9_flag,
            DSM5Assessment.mdd_assessment,
            DSM5Assessment.created_at
        )
        .select_from(User)
        .filter(User.user_id == user_id)
        .join(Phq9Assessment, User.user_id == Phq9Assessment.user_id)
        .join(DSM5Assessment, Phq9Assessment.id == DSM5Assessment.id)
        .order_by(Phq9Assessment.submitted_at)
        .all()
    )
    return results

 
def replace_username_with_user_id(df):
    session = db_engine()
    username = df['username'].iloc[0]
    user_id = session.query(User.user_id).filter(User.username == username).first()
    df['user_id'] = user_id[0]
    df.drop('username', axis=1, inplace=True)
 
    return df
 
 
def upsert_predictions(df, session, phq9_assessment_id: int):
    """
    Upsert predictions from a DataFrame into the predictions table.
 
    If a record (user_id + consultation_seq) exists → update it.
    Otherwise → insert new record.
 
    Args:
        df (pd.DataFrame): DataFrame containing prediction data.
        session (Session): SQLAlchemy session.
        phq9_assessment_id (int): Foreign key to Phq9Assessment table.
    """
    for _, row in df.iterrows():
        existing_record = (
            session.query(Prediction)
            .filter(
                Prediction.user_id == int(row['user_id']),
                Prediction.consultation_seq == int(row['consultation_seq'])
            )
            .first()
        )
 
        if existing_record:
            import math

            if row['phq9_total_score'] is None or (isinstance(row['phq9_total_score'], float) and math.isnan(row['phq9_total_score'])):
                # Handle nan
                existing_record.phq9_total_score = None
            else:
                existing_record.phq9_total_score = int(row['phq9_total_score'])

            print(type(row['phq9_total_score']), " + ", row['phq9_total_score'])
            
            existing_record.relapse = float(row['relapse'])
            existing_record.dsm5_mdd_assessment_enc = float(row['dsm5_q9_flag_enc'])
            existing_record.is_predicted = bool(row['is_predicted'])
            existing_record.phq9_assessment_id = phq9_assessment_id
 
        else:
            # Insert new record
            new_pred = Prediction(
                user_id = int(row['user_id']),
                phq9_total_score = int(row['phq9_total_score']),
                relapse = float(row['relapse']),
                dsm5_mdd_assessment_enc = float(row['dsm5_q9_flag_enc']),
                consultation_seq = int(row['consultation_seq']),
                is_predicted = bool(row['is_predicted']),
                phq9_assessment_id = phq9_assessment_id
            )
            session.add(new_pred)
 
    session.commit()
 
def save_prediction(df, phq9_assessment_id):
    session = db_engine()
    upsert_predictions(df, session, phq9_assessment_id)
    session.close()
   
def extract_pandas_dataframe(limit_users, user_id = None):
    session = db_engine()
    if user_id:
        results = extract_user_data(session, user_id)
    else:
        results = extract_data(session, limit_users)
    session.close()
 
    columns = [
        'username', 'age', 'gender', 'industry', 'profession',
        'phq9_responses', 'phq9_total_score', 'phq9_doctors_notes', 'phq9_patients_notes',
        'dsm5_severity', 'dsm5_q9_flag', 'dsm5_mdd_assessment', 'dsm5_created_at'
    ]
    df = pd.DataFrame(results, columns=columns)
 
    # Count of records per username
    records_per_user = df['username'].value_counts()
 
    print(f"""
    ================= DataFrame Summary =================
    Total Rows: {len(df)}
    Shape (rows, columns): {df.shape}
    Distinct Users: {df['username'].nunique()}
 
    ----------------- Records per User -------------------
    {records_per_user}
 
    ----------------- Descriptive Statistics --------------
    df.describe(include='all')
    ======================================================
    """)
 
    return df
 
def expand_df_convert_dtypes(df):
   
    # Explode phq9 responses
    phq9_df = phq9_df = df["phq9_responses"].apply(pd.Series)
    phq9_df = phq9_df.add_prefix("phq9_q")
    df_expanded = pd.concat([df, phq9_df], axis = 1)
 
    # Datatype conversion
    df_expanded['age'] = df_expanded['age'].astype(int)
    df_expanded['phq9_total_score'] = df_expanded['phq9_total_score'].astype(int)
    df_expanded["dsm5_mdd_assessment"] = (
        df_expanded["dsm5_mdd_assessment"]
        .astype(str)
        .str.strip()            # remove whitespace
        .str.strip("'")         # remove single quotes
        .str.lower()
        .map({'true': True, 'false': False})
        .astype('bool')
    )
    df_expanded['dsm5_created_at'] = pd.to_datetime(df_expanded['dsm5_created_at'], utc=True)
    df_expanded[['age', 'phq9_total_score', 'dsm5_created_at','dsm5_mdd_assessment']].dtypes
 
    return df_expanded
 
def label_encode_columns(df, save_dir = None):
    if save_dir is None:
        save_dir = Config.ml_path('Labels')
    
    Config.ensure_dir(save_dir)
 
    # Encode categorical columns
    label_encoders = {}
 
    # Columns to encode
    categorical_cols = ['industry', 'profession', 'dsm5_severity', 'gender']
 
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        # Save label mapping to CSV
        label_df = pd.DataFrame({
            col: le.classes_,
            col + '_enc': range(len(le.classes_))
        })
        encoder_path = os.path.join(save_dir, f'label_encoder_{col}.pkl')
        Config.ensure_parent(encoder_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
 
    # For boolean columns, convert to int (already bool, so just astype)
    df['dsm5_q9_flag_enc'] = df['dsm5_q9_flag'].astype(int)
    df['dsm5_mdd_assessment_enc'] = df['dsm5_mdd_assessment'].astype(int)
    df[[col + '_enc' for col in categorical_cols] + ['dsm5_q9_flag_enc', 'dsm5_mdd_assessment_enc']].head()
 
    # --- NEW LOGIC: remove original columns if encoded versions exist ---
    for col in df.columns.copy():  # copy to avoid modification during iteration
        enc_col = col + '_enc'
        if col in df.columns and enc_col in df.columns:
            df.drop(columns=[col], inplace=True)
 
    return df
 
def simulate_dates(df):
    """
    Simulate sequential consultation dates for each user based on their dsm5_created_at order.
 
    Logic:
    - Sort by username and dsm5_created_at (millisecond differences determine consultation order)
    - For each user (username):
        * Assign simulated dates in the same chronological order
        * Dates are spaced evenly or sequentially within the last year
    """
 
    np.random.seed(42)
 
    # Sort by user and timestamp (crucial for order)
    df = df.sort_values(['username', 'dsm5_created_at']).reset_index(drop=True)
 
    # Create a base date range (last year)
    base_end = pd.Timestamp.now(tz='UTC')
    base_start = base_end - pd.Timedelta(days=365)
 
    sim_dates = []
 
    # Process each user separately
    for username, group in df.groupby('username', sort=False):
        n = len(group)
 
        # Generate n sorted simulated dates for this user
        # Ensure they are in chronological order
        user_dates = pd.to_datetime(
            np.linspace(base_start.value, base_end.value, n)
        ).tz_localize('UTC')
 
        sim_dates.extend(user_dates)
 
    # Assign the generated dates back to the main DataFrame
    df['sim_date'] = sim_dates
 
    return df
 
 
# Relapse Detection
def detect_relapse(df):
    df = df.sort_values(['username', 'sim_date'])
    df['relapse'] = 0
    for user, group in df.groupby('username'):
        scores = group['phq9_total_score'].values
        relapse_flags = np.zeros_like(scores)
        for i in range(2, len(scores)):
            recent_trend = scores[i-3:i]
            # check if trend was decreasing then suddenly increases/stagnates
            if len(recent_trend) >= 3:
                if recent_trend[-3] > recent_trend[-2] > recent_trend[-1]:  # steady improvement
                    continue
                elif recent_trend[-1] >= recent_trend[-2]:  # rise or plateau after improvement
                    relapse_flags[i] = 1
        df.loc[group.index, 'relapse'] = relapse_flags
    return df
 
def normalize_data(df, save_path=None, cols_path=None):
    # Define paths using Config class
    SCALER_DIR = Config.ml_path('scaler')
    
    if save_path is None:
        save_path = os.path.join(SCALER_DIR, 'minmax_scaler.pkl')
    if cols_path is None:
        cols_path = os.path.join(SCALER_DIR, 'numeric_cols.pkl')
    
    # Ensure directories exist
    Config.ensure_dir(SCALER_DIR)
 
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='float64').columns.tolist()
 
    # Save numeric columns list
    Config.ensure_parent(cols_path)
    with open(cols_path, 'wb') as f:
        pickle.dump(numeric_cols, f)
 
    # Normalize all columns in the dataframe
    min_max_scaler = preprocessing.MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
   
    # Save scaler for later use
    with open(save_path, 'wb') as f:
        pickle.dump(min_max_scaler, f)
 
    # separate scaler for target cols
    target_scaler = preprocessing.MinMaxScaler()    # Total score
    df[['phq9_total_score']] = target_scaler.fit_transform(df[['phq9_total_score']])
    target_scaler_path = Config.ml_path('scaler', 'phq9_total_score_scaler.pkl')
    Config.ensure_parent(target_scaler_path)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
 
    target_scaler = preprocessing.MinMaxScaler()    # q9 flag
    df[['dsm5_q9_flag_enc']] = target_scaler.fit_transform(df[['dsm5_q9_flag_enc']])
    target_scaler_path = Config.ml_path('scaler', 'dsm5_q9_flag_enc_scaler.pkl')
    Config.ensure_parent(target_scaler_path)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
 
    target_scaler = preprocessing.MinMaxScaler()    # relapse
    df[['relapse']] = target_scaler.fit_transform(df[['relapse']])
    target_scaler_path = Config.ml_path('scaler', 'relapse_scaler.pkl')
    Config.ensure_parent(target_scaler_path)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)    
 
    return df_normalized
 
def perform_sentiment_analysis(df):
    df['patients_sentiment'] = df['phq9_patients_notes'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
    df['doctors_sentiment'] = df['phq9_doctors_notes'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
 
    return df
 
 
def _simple_keyword_score(text, keywords):
    text = str(text).lower()
    return int(any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords))
 
def extract_keywords(df, patient_col=None, doctor_col=None):
    cols = df.columns.str.lower()
    if patient_col is None:
        patient_col = next((c for c in df.columns if "patient" in c.lower() and "note" in c.lower()), None)
    if doctor_col is None:
        doctor_col = next((c for c in df.columns if "doctor" in c.lower() and "note" in c.lower()), None)
 
    if patient_col is None or doctor_col is None:
        raise KeyError("Specify patient_col and doctor_col if auto-detection fails.")
 
    # Fast keyword presence detection (1 if found, 0 if not)
    patient_features = {}
    for sym, kws in DSM5_KEYWORDS.items():
        patient_features[sym] = df[patient_col].apply(lambda x: _simple_keyword_score(x, kws))
    patient_df = pd.DataFrame(patient_features)
 
    doctor_features = {}
    for sym, kws in DSM5_KEYWORDS.items():
        doctor_features[sym] = df[doctor_col].apply(lambda x: _simple_keyword_score(x, kws))
    doctor_df = pd.DataFrame(doctor_features)
 
    return patient_df, doctor_df
 
 
def extract_keyword_features_from_notes(df, patient_col="phq9_patients_notes", doctor_col="phq9_doctors_notes"):
    patient_df, doctor_df = extract_keywords(df, patient_col, doctor_col)
    patient_df = patient_df.add_prefix("patient_")
    doctor_df = doctor_df.add_prefix("doctor_")
   
    # Drop original note columns
    df_out = df.drop(columns=[patient_col, doctor_col], errors='ignore')
 
    return pd.concat([df_out.reset_index(drop=True), patient_df, doctor_df], axis=1)
 
def correlation_analysis(df, threshold=0.2, always_include=None):
    """
    Returns a DataFrame with columns that are highly correlated above the threshold,
    plus any columns in always_include.
    """
    if always_include is None:
        always_include = ['username', 'sim_date', 'phq9_total_score']
 
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()
 
    # Compute absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()
 
    # Upper triangle to avoid duplicate pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
 
    # Find columns with correlations above threshold
    highly_correlated = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
 
    # Combine with always_include columns if they exist in df
    cols_to_return = [col for col in always_include if col in df.columns] + highly_correlated
 
    return df[cols_to_return], highly_correlated
 
 
def extract_correlating_columns(df, threshold=0.8, always_include=None):
    correlating_df, correlating_cols = correlation_analysis(df, threshold, always_include)
 
    # Keep always_include columns + correlating columns
    if always_include is None:
        always_include = ['username', 'sim_date', 'phq9_total_score']
    cols_to_keep = [col for col in always_include if col in correlating_df.columns] + correlating_cols
 
    # Remove duplicates while keeping order
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
 
    return correlating_df[cols_to_keep], cols_to_keep
 
 
 
 
def extract_cleaned_dataframe(limit_users=100, mode="train", user_id=None):
    """
    Extract and preprocess the dataframe.
   
    mode: "train" or "inference"
        - In train mode, highly correlating columns are saved.
        - In inference mode, previously saved columns are used to ensure consistent columns.
    """
    if user_id:
        df = extract_pandas_dataframe(limit_users, user_id)
    else:
        df = extract_pandas_dataframe(limit_users)
 
    df_expanded = expand_df_convert_dtypes(df)
 
    # Select relevant columns
    selected_columns = [
        'username', 'age', 'gender', 'industry', 'profession',
        'phq9_total_score', 'phq9_doctors_notes', 'phq9_patients_notes',
        'dsm5_severity', 'dsm5_q9_flag', 'dsm5_mdd_assessment', 'dsm5_created_at',
        'phq9_q1', 'phq9_q2', 'phq9_q3', 'phq9_q4', 'phq9_q5', 'phq9_q6', 'phq9_q7', 'phq9_q8','phq9_q9'
    ]
    df_expanded = df_expanded[selected_columns]
 
    # Encode categorical columns
    df_encoded = label_encode_columns(df_expanded)
 
    # Simulate dates
    df = simulate_dates(df_encoded)
 
    # Detect relapse
    df = detect_relapse(df)
 
    # Sentiment analysis
    df = perform_sentiment_analysis(df)
 
    # Keyword features
    df = extract_keyword_features_from_notes(df)
 
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_cols]
    df_normalized = normalize_data(df_numeric)
    df[numeric_cols] = df_normalized
    CORR_COLS_FILE = Config.ml_path('correlations', 'highly_correlated_columns.pkl')
    Config.ensure_parent(CORR_COLS_FILE)
    # Extract highly correlating columns
    if mode == "train":
        df_corr, highly_correlated_cols = extract_correlating_columns(df, threshold=0.2)
        # Save the column names for future use
        joblib.dump(highly_correlated_cols, CORR_COLS_FILE)
    else:  # inference mode
        if not os.path.exists(CORR_COLS_FILE):
            raise FileNotFoundError(f"Correlated columns file not found: {CORR_COLS_FILE}")
        highly_correlated_cols = joblib.load(CORR_COLS_FILE)
        # Filter df to only include these exact columns
        df_corr = df[highly_correlated_cols].copy()
 
    # Always include 'username' and 'sim_date' if present
    final_cols = highly_correlated_cols
    final_cols = [c for c in final_cols if c in df.columns]  # safety check
    df_final = df[final_cols].copy()
 
    return df_final
 
def extract_dataframe_for_training(limit_users = 100):
    df = extract_cleaned_dataframe(limit_users, mode = 'train')
    return df
 
def extract_dataframe_for_testing(limit_users=1):
    df = extract_cleaned_dataframe(limit_users, mode = 'test')
    return df
 
def extract_dataframe_of_user(user_id):
    df = extract_cleaned_dataframe(limit_users=1, mode = 'test', user_id=user_id)
    return df
 
# if __name__ == "__main__":
#     df = extract_cleaned_dataframe(limit_users=100, mode = 'train').head(30)
 
#     import pprint
 
#     pprint.pprint(df.iloc[1].to_dict())
 
#     print(len(df))
 
# if __name__ == "__main__":
#     df = extract_dataframe_of_user(user_id = 1)
#     print(df.to_string())
 