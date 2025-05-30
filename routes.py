from flask import request, jsonify, render_template_string, send_from_directory
from predictor import StrokePredictor
from gemini import get_gemini_analysis
from config import GEMINI_API_KEY, GEMINI_API_URL
import logging
import os
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

predictor = StrokePredictor()

def register_routes(app):
    if not predictor.load_model():
        logger.info("No pre-trained model found. Use /train endpoint to train a new model.")

    @app.route('/')
    def home():
        """Serve the main HTML page"""
        try:
            with open('index.html', 'r', encoding='utf-8') as f:
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
                        if (result.gemini_analysis) {
                            displayGeminiAnalysis(result.gemini_analysis);
                        }
                    } else {
                        alert('Prediction failed: ' + result.error);
                    }
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while making the prediction.');
                    document.getElementById('loading').style.display = 'none';
                });
                return;
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
                'error': 'index.html not found',
                'message': 'Please ensure index.html is in the frontend directory'
            }), 404

    @app.route('/frontend/<path:filename>')
    def serve_frontend_files(filename):
        """Serve static files from frontend folder"""
        return send_from_directory('frontend', filename)

    @app.route('/predict', methods=['POST'])
    def predict_stroke():
        """API endpoint for stroke prediction with Gemini analysis"""
        try:
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