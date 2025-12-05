from calendar import c
import json
import pdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from services.data_process_and_model_training import (
    extract_dataframe_for_training,
    extract_dataframe_for_testing,
    extract_dataframe_of_user,
    replace_username_with_user_id,
    save_prediction
    )
from config import Config
import joblib
import os
import pickle
import shap
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, explained_variance_score
import numpy as np


# Model paths from Config
MODEL_BASE_PATH = Config.ml_path('Models')
MODEL_PATH = os.path.join(MODEL_BASE_PATH, 'final_patient_lstm.keras')
Config.ensure_parent(MODEL_PATH)
 
def pad_2d_sequence(seq, max_t):
    """
    Pads or truncates a 2D numpy array (timesteps × features)
    to a fixed number of timesteps (max_t).
    Pads with zeros at the start if shorter.
    """
    seq = np.asarray(seq, dtype='float32')
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
 
    n_t, n_f = seq.shape
    if n_t >= max_t:
        # Keep the most recent max_t timesteps
        return seq[-max_t:, :]
   
    # Pre-pad with zeros (older consultations first)
    pad = np.zeros((max_t - n_t, n_f), dtype='float32')
    return np.vstack([pad, seq])
 
 
def detect_relapse(df):
    df = df.sort_values(['username', 'sim_date'])
    df['relapse'] = 0
 
    for user, group in df.groupby('username'):
        scores = group['phq9_total_score'].values
        relapse_flags = np.zeros_like(scores)
 
        for i in range(2, len(scores)):
            recent_trend = scores[max(0, i-3):i+1]
            if len(recent_trend) >= 3:
                if recent_trend[-1] >= recent_trend[-2] and recent_trend[-2] < recent_trend[-3]:
                    relapse_flags[i] = 1
 
        df.loc[group.index, 'relapse'] = relapse_flags
 
    return df
 
def predict_future_consultations(model, patient_df, feature_cols, target_cols, max_t=21):
    """
    Predict future consultations for a single patient or multiple patients.
    """
    X_patient = patient_df[feature_cols].to_numpy()
 
    # Pad sequence to max_t
    # X_padded = pad_sequences([X_patient], maxlen=max_t, dtype='float32', padding='pre')
 
    X_padded, Y_padded = prepare_sequences(patient_df, feature_cols, target_cols, max_t)
 
    # Predict full timeline
    Y_pred = model.predict(X_padded, verbose=0)

    dump_dir = Config.ml_path('model_dump')
    Config.ensure_dir(dump_dir)



    completed = len(patient_df)
    remaining = max_t - completed
 
    if remaining <= 0:
        return Y_pred, pd.DataFrame()
 
    # Create DataFrame for predicted consultations
    df_pred = pd.DataFrame(Y_pred[0, -remaining:, :], columns=target_cols)
    df_pred['consultation_seq'] = np.arange(completed + 1, completed + 1 + remaining)
 
    return Y_pred, df_pred
 
 
def prepare_sequences(df, feature_cols, target_cols, max_t=21):
    """
    Prepare padded sequences per patient.
    df: cleaned DataFrame sorted by username and sim_date
    feature_cols: list of feature column names
    target_cols: list of target column names (multivariate)
    max_t: total number of consultations per patient to predict
    """
    X_seqs = []
    Y_seqs = []
   
    df_sorted = df.sort_values(['username', 'sim_date'])
    df_sorted['consultation_seq'] = df_sorted.groupby('username').cumcount() + 1
    df_sorted['days_since_last'] = df_sorted.groupby('username')['sim_date'].diff().dt.days.fillna(0)
   
    for user, group in df_sorted.groupby('username'):
        X_user = group[feature_cols].to_numpy()
        Y_user = group[target_cols].to_numpy()
       
        # # Pad sequences to max_t
        # X_padded = pad_sequences([X_user], maxlen=max_t, dtype='float32', padding='pre')[0]
        # Y_padded = pad_sequences([Y_user], maxlen=max_t, dtype='float32', padding='pre')[0]
 
        # ✅ FIXED: use safe 2D padding instead of keras.pad_sequences
        X_padded = pad_2d_sequence(X_user, max_t)
        Y_padded = pad_2d_sequence(Y_user, max_t)
 
        X_seqs.append(X_padded)
        Y_seqs.append(Y_padded)
   
    return np.array(X_seqs), np.array(Y_seqs)
 
def build_lstm_model(num_features, num_targets, max_t=21, lstm_units=64):
    """
    Build a multi-step, multivariate LSTM model
    """
    encoder_inputs = Input(shape=(max_t, num_features))
    masked = Masking(mask_value=0.)(encoder_inputs)
    lstm_out = LSTM(lstm_units, return_sequences=True)(masked)
    outputs = TimeDistributed(Dense(num_targets))(lstm_out)
   
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
 
# def train_patient_lstm(df, feature_cols, target_cols, max_t=21, lstm_units=64, epochs=20, batch_size=32):
#     """
#     Functional pipeline:
#     - Prepare sequences
#     - Build LSTM
#     - Train model
#     - Save model
#     Returns: trained model, X_array, Y_array
#     """
#     X_array, Y_array = prepare_sequences(df, feature_cols, target_cols, max_t)
   
#     model = build_lstm_model(num_features=X_array.shape[2],
#                              num_targets=Y_array.shape[2],
#                              max_t=max_t,
#                              lstm_units=lstm_units)
#     joblib.dump(feature_cols, "feature_cols.pkl")
#     joblib.dump(target_cols, "target_cols.pkl")
   
#     checkpoint = ModelCheckpoint(MODEL_PATH, monitor='loss', save_best_only=True, verbose=1)
   
#     model.fit(X_array, Y_array,
#               epochs=epochs,
#               batch_size=batch_size,
#               callbacks=[checkpoint])
#     model.summary()
#     model.save(MODEL_PATH, save_format="keras")
   
#     print(f"Model saved at {MODEL_PATH}")
#     return model, X_array, Y_array
 

def train_patient_lstm(df, feature_cols, target_cols, max_t=21, lstm_units=64, epochs=20, batch_size=32):
    """
    Functional pipeline:
    - Prepare sequences
    - Build LSTM
    - Train model
    - Evaluate model
    - Generate plots
    - Compute feature importance
    - Save results to DB
 
    Returns: trained model, X_array, Y_array
    """
 
    # === Prepare sequences ===
    X_array, Y_array = prepare_sequences(df, feature_cols, target_cols, max_t)
    print(f"Prepared sequences: X={X_array.shape}, Y={Y_array.shape}")
 
    # === Train/test split ===
    X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
 
    # === Build LSTM model ===
    model = build_lstm_model(
        num_features=X_array.shape[2],
        num_targets=Y_array.shape[2],
        max_t=max_t,
        lstm_units=lstm_units
    )
 
    # Save feature & target columns for later
    # Define model dump paths using Config
    MODEL_DUMP_DIR = Config.ml_path('model_dump')
    FEATURE_COLS_PATH = os.path.join(MODEL_DUMP_DIR, "feature_cols.pkl")
    TARGET_COLS_PATH = os.path.join(MODEL_DUMP_DIR, "target_cols.pkl")
    
    # Ensure directories exist
    Config.ensure_dir(MODEL_DUMP_DIR)
    
    # Save the files
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    joblib.dump(target_cols, TARGET_COLS_PATH)
 
    # Model checkpoint
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    # === Train model ===
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=1
    )
 
    # model.save(MODEL_PATH, save_format="keras")
    print("✅ Model saved at final_patient_lstm.keras")


    # === Evaluate model ===
    Y_pred = model.predict(X_test)
    Y_test_flat = Y_test.reshape(-1, Y_test.shape[-1])
    Y_pred_flat = Y_pred.reshape(-1, Y_pred.shape[-1])
 

    metrics_dict = {
        "mse": float(mean_squared_error(Y_test_flat, Y_pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(Y_test_flat, Y_pred_flat))),
        "mae": float(mean_absolute_error(Y_test_flat, Y_pred_flat)),
        "mape": float(mean_absolute_percentage_error(Y_test_flat, Y_pred_flat)),
        "r2": float(r2_score(Y_test_flat, Y_pred_flat)),
        "explained_variance": float(explained_variance_score(Y_test_flat, Y_pred_flat))
    }
 
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
 
    # === Plot predictions with Seaborn ===

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
 
    graph_paths = []
    try:
        for i, target in enumerate(target_cols):
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=Y_test_flat[:, i], y=Y_pred_flat[:, i], alpha=0.6, s=50)
            sns.lineplot(x=Y_test_flat[:, i], y=Y_test_flat[:, i], color="red", label="Ideal Fit")
            plt.title(f"{target} - Predicted vs Actual", fontsize=14, weight="bold")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.tight_layout()
            plots_dir = Config.ml_path('model_dump', 'plots')
            Config.ensure_dir(plots_dir)
            path = os.path.join(plots_dir, f"{target}_pred_vs_actual.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_paths.append(path)
            print(f"📊 Saved plot: {path}")
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")
 
    # === Feature importance (SHAP) ===
    try:
        sample_X = X_train[:100] if X_train.shape[0] > 100 else X_train
        explainer = shap.DeepExplainer(model, sample_X)
        shap_values = explainer.shap_values(X_test[:100])
        feature_importance = np.mean(np.abs(shap_values[0]), axis=(0, 1))
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        feature_importance_dict = feature_importance_df.to_dict(orient="records")
        print("\n=== Top Features ===")
        print(feature_importance_df.head(10))
    except Exception as e:
        print(f"⚠️ Feature importance failed: {e}")
        feature_importance_dict = []
 
    # === Save results to DB (placeholder function) ===
    try:
        results_dict = {
            "model_name": "patient_lstm",
            "metrics": metrics_dict,
            "feature_importance": feature_importance_dict,
            "plots": graph_paths,
            "model_name": "patient_lstm"
        }
 
        # Save model evaluation metrics to .json
        dump_dir = Config.ml_path('model_dump')
        Config.ensure_dir(dump_dir)
        results_json_path = os.path.join(dump_dir, "model_evaluation_results.json")
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"✅ Model evaluation metrics saved to {results_json_path}")
        print("✅ Evaluation Results saved successfully.")
    except Exception as e:
        print(f"⚠️ Could not save results to DB: {e}")
 
    return model, X_array, Y_array
 

def prepare_features_and_targets(df):
    # ---- Target Columns ----
    target_cols = ['dsm5_q9_flag_enc', 'relapse', 'phq9_total_score']
 
    # ---- Non-feature columns (drop if exist) ----
    remove_cols = [
        'username', 'phq9_doctors_notes', 'phq9_patients_notes',
        'dsm5_created_at', 'sim_date'
    ]
 
    # ---- Keep only existing columns ----
    exclude_cols = set([col for col in remove_cols if col in df.columns] +
                       [col for col in target_cols if col in df.columns])
 
    # ---- Dynamically select features ----
    feature_cols = [col for col in df.columns if col not in exclude_cols]
 
    return feature_cols, target_cols
 
def train_lstm_model_for_timeline_prediction(limit_users=10):
    # 1. Extract dataframe
    df = extract_dataframe_for_training(limit_users)
    # 2. Detect relapse
    df = detect_relapse(df)
    # 3. Prepare features and targets dynamically
    feature_cols, target_cols = prepare_features_and_targets(df)
    # 4. Train model
    model, X_array, Y_array = train_patient_lstm(df, feature_cols, target_cols, max_t=21)
    return model, feature_cols, target_cols
 
def predict_future_timeline(model, patient_df, feature_cols, target_cols, max_t=21):
    """
    Wrapper that checks number of consultations before predicting future ones.
    Combines past + predicted consultations into one timeline DataFrame.
    """
    completed_consultations = len(patient_df)
    username = patient_df['username'].iloc[0] if 'username' in patient_df.columns else 'Unknown'
 
    if completed_consultations >= max_t:
        msg = f"✅ User '{username}' already has {completed_consultations} consultations — no future prediction needed."
        print(msg)
        timeline_df = patient_df[['username', 'sim_date', 'dsm5_q9_flag_enc', 'relapse', 'phq9_total_score']].copy()
        timeline_df['is_predicted'] = False
        timeline_df = timeline_df.sort_values(['sim_date'], ascending=True)
       
        return np.array([]), timeline_df, msg
 
    msg = f"🔮 Predicting future consultations for user '{username}' ({completed_consultations}/{max_t} completed)..."
    print(msg)
 
    # Predict remaining consultations
    Y_pred, df_future = predict_future_consultations(model, patient_df, feature_cols, target_cols, max_t=max_t)
 
    # Mark and merge
    df_future['username'] = username
    df_future['is_predicted'] = True
 
    # Copy actual data
    patient_df_copy = patient_df[['username', 'sim_date', 'dsm5_q9_flag_enc', 'relapse', 'phq9_total_score']].copy()
    patient_df_copy['is_predicted'] = False
    patient_df_copy = patient_df_copy.sort_values(['sim_date'], ascending=True)
    patient_df_copy['consultation_seq'] = range(1, completed_consultations+1)
 
    # Combine both
    timeline_df = pd.concat([patient_df_copy, df_future], ignore_index=True)
    timeline_df = timeline_df.sort_values('consultation_seq', ignore_index=True)

 
    return timeline_df, msg
 
 
def reverse_transform_df(df, label_dir='Labels', scaler_path=None, cols_path=None):
    if scaler_path is None:
        scaler_path = Config.ml_path('scaler', 'minmax_scaler.pkl')
    if cols_path is None:
        cols_path = Config.ml_path('scaler', 'numeric_cols.pkl')
 
    target_scaler_path = Config.ml_path('scaler', 'phq9_total_score_scaler.pkl')
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
 
        if 'phq9_total_score' in df.columns:
            df[['phq9_total_score']] = target_scaler.inverse_transform(df[['phq9_total_score']])
   
    target_scaler_path = Config.ml_path('scaler', 'dsm5_q9_flag_enc_scaler.pkl')
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
 
        if 'dsm5_q9_flag_enc' in df.columns:
            df[['dsm5_q9_flag_enc']] = target_scaler.inverse_transform(df[['dsm5_q9_flag_enc']])
 
    target_scaler_path = Config.ml_path('scaler', 'relapse_scaler.pkl')
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
 
        if 'relapse' in df.columns:
            df[['relapse']] = target_scaler.inverse_transform(df[['relapse']])
 
    return df
 
def predict_user_timeline(user_id, phq9_assessment_id):
    model = load_model(MODEL_PATH)
    df = extract_dataframe_of_user(user_id)
    feature_cols, target_cols = prepare_features_and_targets(df)
    df = detect_relapse(df)
    df = df.sort_values(['username', 'sim_date']).reset_index(drop=True)
    timeline_df, msg = predict_future_timeline(model, df, feature_cols, target_cols)
    timeline_df_original = reverse_transform_df(timeline_df)
    timeline_df = replace_username_with_user_id(timeline_df_original)

    # print("Columns:", timeline_df.columns.tolist())
    timeline_df['relapse'] = timeline_df['relapse'].round(0).clip(0, 1).abs()
    timeline_df['dsm5_q9_flag_enc'] = timeline_df['dsm5_q9_flag_enc'].round(0).clip(0, 1).abs()
 
    save_prediction(timeline_df, phq9_assessment_id)

    timeline_dict = timeline_df.to_dict(orient="records")

    return timeline_dict
    # print(timeline_df_original.to_string(index=False, float_format='%.6f'))  # last few consultations (actual + predicted)
 
    # write_predictions_to_db(df)
 
# if __name__ == "__main__":
#     # model, feature_cols, target_cols = train_lstm_model_for_timeline_prediction(limit_users=10000)
 
#     model = load_model(MODEL_PATH)
#     df = extract_dataframe_for_testing(limit_users=1)
#     print("-" * 100)
#     print(df.to_string(index=False, float_format='%.6f'))  # last few consultations (actual + predicted)
#     print("-" * 100)
 
#     feature_cols, target_cols = prepare_features_and_targets(df)
#     df = detect_relapse(df)
#     df = df.sort_values(['username', 'sim_date']).reset_index(drop=True)
#     print(df.to_string(index=False, float_format='%.6f'))  # last few consultations (actual + predicted)
#     print("-" * 100)
#     print("***extracted_cols***", df.columns)
#     timeline_df, msg = predict_future_timeline(model, df, feature_cols, target_cols)
#     print(msg)
#     print("timeline_df\n", timeline_df.to_string(index=False, float_format='%.6f'))  # last few consultations (actual + predicted)
#     print("-" * 100)
 
#     # Reverse transform predicted timeline
#     timeline_df_original = reverse_transform_df(timeline_df)
#     # print(timeline_df_original.tail())
#     print(timeline_df_original.to_string(index=False, float_format='%.6f'))  # last few consultations (actual + predicted)
 
# if __name__=="__main__":
#     predict_user_timeline("6198", "228164")
 
 
 
 
 
 
 
 
# if __name__ == "__main__":
#     train_lstm_model_for_timeline_prediction()
 
# if __name__ == "__main__":
#     # Train LSTM
#     model, feature_cols, target_cols = train_lstm_model_for_timeline_prediction(100)
 
#     # Predict future timeline for a specific patient
#     patient_df = extract_dataframe_for_training(limit_users=50)
#     patient_df = patient_df[patient_df['username'] == 'teresa.duncan685'].sort_values('sim_date')
 
#     Y_pred, df_future = predict_future_timeline(model, patient_df, feature_cols, target_cols)
#     print(df_future)
 
 
 
 
 