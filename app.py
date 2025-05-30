from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
import traceback
from dotenv import load_dotenv
from gemini import get_gemini_analysis

try:
    from ModelTraining import StrokePredictionTrainer
except ImportError:
    print("Warning: ModelTraining.py not found. Model training functionality will be limited.")

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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
            
            # Run the complete training pipeline
            best_model, best_model_name, best_accuracy = self.trainer.run_complete_pipeline(perform_tuning=True)
            
            # Load the newly trained model
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
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Handle BMI if it's N/A or missing
            if 'bmi' in df.columns:
                df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
                if df['bmi'].isna().any():
                    df['bmi'] = df['bmi'].fillna(28.0)  # Default BMI
            
            # Convert numeric fields
            numeric_fields = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # Apply label encoding if encoders are available
            if self.label_encoders:
                for col, encoder in self.label_encoders.items():
                    if col in df.columns:
                        try:
                            # Handle the specific case where residence_type might be 'residence_type' in form but 'Residence_type' in model
                            if col == 'Residence_type' and 'residence_type' in patient_data:
                                df['Residence_type'] = df.pop('residence_type') if 'residence_type' in df.columns else patient_data['residence_type']
                            
                            df[col] = encoder.transform(df[col].astype(str))
                        except ValueError as e:
                            # Handle unseen labels
                            logger.warning(f"Unseen label in {col}: {df[col].values[0] if not df[col].empty else 'empty'}")
                            # Use the most common class (usually 0)
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
            # Map frontend field names to model field names if needed
            mapped_data = patient_data.copy()
            if 'residence_type' in mapped_data:
                mapped_data['Residence_type'] = mapped_data.pop('residence_type')
            
            # Preprocess the input
            processed_data = self.preprocess_input(mapped_data)
            
            # Scale data if scaler is available and needed
            # Tree-based models don't need scaling
            tree_based_models = ['RandomForestClassifier', 'DecisionTreeClassifier', 
                               'GradientBoostingClassifier', 'AdaBoostClassifier']
            
            model_name = type(self.model).__name__
            
            if model_name in tree_based_models:
                X = processed_data.values
            elif self.scaler is not None:
                X = self.scaler.transform(processed_data.values)
            else:
                X = processed_data.values
            
            # Make prediction
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


predictor = StrokePredictor()

if not predictor.load_model():
    logger.info("No pre-trained model found. You can train a new model using the /train endpoint.")

@app.route('/')
def home():
    """Serve the main HTML page"""
    try:
        with open('Frontend.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        modified_html = html_content.replace(
            "// Mock ML model - In production, this would connect to your Python backend",
            "// Connected to Flask backend"
        ).replace(
            "const result = model.predict(patientData);",
            """
            // Send data to Flask backend
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(patientData)
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    displayResults(result.data);
                    // Show Gemini analysis if available
                    if (result.gemini_analysis) {
                        displayGeminiAnalysis(result.gemini_analysis);
                    }
                } else {
                    alert('Prediction failed: ' + result.error);
                }
                // Hide loading and show results
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while making the prediction.');
                document.getElementById('loading').style.display = 'none';
            });
            return; // Exit early since we're using async calls
            """
        ).replace(
            "// Make prediction\n            const result = model.predict(patientData);",
            "// This code block is now handled by the fetch request above"
        ).replace(
            "// Display results\n                displayResults(result);",
            "// Results are displayed in the fetch callback"
        )
        
        return render_template_string(modified_html)
        
    except FileNotFoundError:
        return jsonify({
            'error': 'Frontend.html not found',
            'message': 'Please ensure Frontend.html is in the same directory as this Flask app'
        }), 404

@app.route('/predict', methods=['POST'])
def predict_stroke():
    """API endpoint for stroke prediction with Gemini analysis"""
    try:
        # Get JSON data from request
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'message': 'Please provide patient data in JSON format'
            }), 400
        
        logger.info(f"Received prediction request: {patient_data}")
        
        result = predictor.predict(patient_data)
        
        logger.info(f"Prediction result: {result}")
        
        gemini_response = get_gemini_analysis(patient_data, result, GEMINI_API_KEY, GEMINI_API_URL)
        
        return jsonify({
            'success': True,
            'data': result,
            'gemini_analysis': gemini_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        logger.error(f"ValueError in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Model not loaded or invalid input data'
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An unexpected error occurred during prediction'
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """API endpoint to train a new model"""
    try:
        data = request.get_json()
        csv_file_path = data.get('csv_file_path', 'data/healthcare-dataset-stroke-data.csv')
        
        if not os.path.exists(csv_file_path):
            return jsonify({
                'success': False,
                'error': f'Dataset file not found: {csv_file_path}',
                'message': 'Please provide a valid path to the CSV dataset'
            }), 400
        
        logger.info(f"Starting model training with dataset: {csv_file_path}")
        
        # Train new model
        result = predictor.train_new_model(csv_file_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Model training completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': result.get('message', 'Model training failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during model training'
        }), 500

@app.route('/model/status')
def model_status():
    """Get current model status"""
    return jsonify({
        'is_loaded': predictor.is_loaded,
        'model_type': type(predictor.model).__name__ if predictor.model else None,
        'has_scaler': predictor.scaler is not None,
        'has_encoders': predictor.label_encoders is not None,
        'encoder_columns': list(predictor.label_encoders.keys()) if predictor.label_encoders else [],
        'gemini_api_configured': GEMINI_API_KEY is not None
    })

@app.route('/model/reload')
def reload_model():
    """Reload the model from saved files"""
    try:
        success = predictor.load_model()
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to reload model'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error reloading model'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor.is_loaded,
        'gemini_configured': GEMINI_API_KEY is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True) 
    if not predictor.is_loaded:
        print("⚠️  No model loaded. Use /train endpoint to train a new model.")
    if not GEMINI_API_KEY:
        print("⚠️  Gemini API key not set. Set GEMINI_API_KEY environment variable.")
    print("="*80)
    app.run(debug=True, host='0.0.0.0', port=5000)