import json
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gemini_analysis(patient_data, prediction_result, GEMINI_API_KEY, GEMINI_API_URL):
    """Send prediction results to Gemini for analysis and recommendations"""
    if not GEMINI_API_KEY:
        return {
            'success': False,
            'error': 'Gemini API key not configured',
            'analysis': 'Gemini analysis unavailable - API key not set'
        }
    
    try:
        prompt = f"""
        As a medical AI assistant, please analyze the following stroke risk prediction results and provide professional medical insights:

        Patient Information:
        - Age: {patient_data.get('age', 'N/A')}
        - Gender: {patient_data.get('gender', 'N/A')}
        - Hypertension: {'Yes' if patient_data.get('hypertension') == '1' else 'No'}
        - Heart Disease: {'Yes' if patient_data.get('heart_disease') == '1' else 'No'}
        - Ever Married: {patient_data.get('ever_married', 'N/A')}
        - Work Type: {patient_data.get('work_type', 'N/A')}
        - Residence Type: {patient_data.get('residence_type', 'N/A')}
        - Average Glucose Level: {patient_data.get('avg_glucose_level', 'N/A')} mg/dL
        - BMI: {patient_data.get('bmi', 'N/A')}
        - Smoking Status: {patient_data.get('smoking_status', 'N/A')}

        Prediction Results:
        - Risk Level: {prediction_result['risk_level']}
        - Stroke Probability: {prediction_result['probability_stroke']:.1%}
        - No Stroke Probability: {prediction_result['probability_no_stroke']:.1%}

        Please provide:
        1. A brief analysis of the key risk factors
        2. Personalized health recommendations
        3. Lifestyle modifications that could help reduce stroke risk
        4. When to seek medical attention

        Keep the response concise, professional, and include appropriate medical disclaimers.
        """

        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                analysis_text = result['candidates'][0]['content']['parts'][0]['text']
                return {
                    'success': True,
                    'analysis': analysis_text
                }
            else:
                return {
                    'success': False,
                    'error': 'No analysis generated',
                    'analysis': 'Unable to generate analysis at this time.'
                }
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return {
                'success': False,
                'error': f'API request failed: {response.status_code}',
                'analysis': 'Unable to connect to Gemini AI service.'
            }
            
    except requests.RequestException as e:
        logger.error(f"Network error calling Gemini API: {str(e)}")
        return {
            'success': False,
            'error': f'Network error: {str(e)}',
            'analysis': 'Unable to connect to Gemini AI service due to network issues.'
        }
    except Exception as e:
        logger.error(f"Unexpected error calling Gemini API: {str(e)}")
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'analysis': 'An unexpected error occurred while generating analysis.'
        }