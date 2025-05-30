import pandas as pd
import joblib
import os
import logging
import traceback

try:
    from ModelTraining import StrokePredictionTrainer
except ImportError:
    print("Warning: ModelTraining.py not found. Model training functionality will be limited.")

logger = logging.getLogger(__name__)

class StrokePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.model_name = None
        self.is_loaded = False
        self.trainer = None
        
    def load_model(self, model_path='best_stroke_model.joblib', 
                   scaler_path='scaler.joblib', 
                   encoders_path='label_encoders.joblib'):
        """Load the trained model and preprocessing objects"""
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
                
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                logger.info(f"Label encoders loaded from {encoders_path}")
            else:
                logger.warning(f"Encoders file not found: {encoders_path}")
                
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def train_new_model(self, csv_file_path):
        """Train a new model using the provided dataset"""
        try:
            logger.info(f"Starting model training with dataset: {csv_file_path}")
            self.trainer = StrokePredictionTrainer(csv_file_path)
            
            best_model, best_model_name, best_accuracy = self.trainer.run_complete_pipeline(perform_tuning=True)
            
            self.load_model()
            
            return {
                'success': True,
                'model_name': best_model_name,
                'accuracy': best_accuracy,
                'message': 'Model trained successfully'
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Model training failed'
            }
    
    def preprocess_input(self, patient_data):
        """Preprocess patient data for prediction"""
        try:
            df = pd.DataFrame([patient_data])
            
            if 'bmi' in df.columns:
                df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
                if df['bmi'].isna().any():
                    df['bmi'] = df['bmi'].fillna(28.0)
            
            numeric_fields = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            
            if self.label_encoders:
                for col, encoder in self.label_encoders.items():
                    if col in df.columns:
                        try:
                            if col == 'Residence_type' and 'residence_type' in patient_data:
                                df['Residence_type'] = df.pop('residence_type') if 'residence_type' in df.columns else patient_data['residence_type']
                            
                            df[col] = encoder.transform(df[col].astype(str))
                        except ValueError as e:
                            logger.warning(f"Unseen label in {col}: {df[col].values[0] if not df[col].empty else 'empty'}")
                            df[col] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(f"Patient data: {patient_data}")
            raise
    
    def predict(self, patient_data):
        """Make prediction for patient data"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        
        try:
            mapped_data = patient_data.copy()
            if 'residence_type' in mapped_data:
                mapped_data['Residence_type'] = mapped_data.pop('residence_type')
            
            processed_data = self.preprocess_input(mapped_data)
            
            tree_based_models = ['RandomForestClassifier', 'DecisionTreeClassifier', 
                               'GradientBoostingClassifier', 'AdaBoostClassifier']
            
            model_name = type(self.model).__name__
            
            if model_name in tree_based_models:
                X = processed_data.values
            elif self.scaler is not None:
                X = self.scaler.transform(processed_data.values)
            else:
                X = processed_data.values
            
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            return {
                'prediction': int(prediction),
                'probability_no_stroke': float(probabilities[0]),
                'probability_stroke': float(probabilities[1]),
                'risk_level': 'HIGH' if prediction == 1 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise