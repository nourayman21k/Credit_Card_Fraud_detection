from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)
print("Template folder:", app.template_folder)
print("Exists:", os.path.exists(app.template_folder))



# Load the trained model and other files
print("Loading model...")
try:
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    model_info = joblib.load('model_info.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model: {model_info['model_name']}")
    print(f"Method: {model_info['method_name']}")
    print(f"Recall: {model_info['recall']:.4f}")
    print(f"F1-Score: {model_info['f1_score']:.4f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have run credit_card_model.py first to generate the .pkl files")

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a transaction is fraudulent
    Expects JSON data with all feature values
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features in the correct order
        features = []
        missing_features = []
        
        for feature in feature_names:
            if feature in data:
                features.append(float(data[feature]))
            else:
                # Default to 0 if feature is missing
                features.append(0.0)
                missing_features.append(feature)
        
        # Log if any features were missing
        if missing_features:
            print(f"⚠️  Missing features (defaulted to 0): {missing_features}")
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Calculate confidence level
        fraud_prob = probability[1]
        if fraud_prob > 0.85 or fraud_prob < 0.15:
            confidence = 'Very High'
        elif fraud_prob > 0.70 or fraud_prob < 0.30:
            confidence = 'High'
        elif fraud_prob > 0.55 or fraud_prob < 0.45:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Prepare response
        result = {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability[1] * 100),
            'legitimate_probability': float(probability[0] * 100),
            'confidence': confidence,
            'model_info': {
                'model_name': model_info['model_name'],
                'method': model_info['method_name'],
                'recall': float(model_info['recall']),
                'f1_score': float(model_info['f1_score']),
                'precision': float(model_info['precision'])
            }
        }
        
        # Log prediction
        print(f"\n{'='*50}")
        print(f"Prediction: {result['prediction']}")
        print(f"Fraud Probability: {result['fraud_probability']:.2f}%")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*50}\n")
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({'error': f'Invalid data format: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple transactions at once
    Expects JSON array of transaction objects
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected array of transactions'}), 400
        
        results = []
        
        for idx, transaction in enumerate(data):
            # Extract features
            features = []
            for feature in feature_names:
                features.append(float(transaction.get(feature, 0)))
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
            results.append({
                'transaction_id': idx + 1,
                'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'fraud_probability': float(probability[1] * 100),
                'legitimate_probability': float(probability[0] * 100)
            })
        
        # Calculate summary statistics
        fraud_count = sum(1 for r in results if r['prediction'] == 'FRAUD')
        
        response = {
            'results': results,
            'summary': {
                'total_transactions': len(results),
                'fraud_detected': fraud_count,
                'legitimate': len(results) - fraud_count,
                'fraud_percentage': (fraud_count / len(results) * 100) if results else 0
            }
        }
        
        print(f"\n{'='*50}")
        print(f"Batch Prediction Summary:")
        print(f"Total: {response['summary']['total_transactions']}")
        print(f"Fraud: {response['summary']['fraud_detected']}")
        print(f"Legitimate: {response['summary']['legitimate']}")
        print(f"{'='*50}\n")
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'model_name': model_info['model_name'],
            'method': model_info['method_name'],
            'recall': float(model_info['recall']),
            'f1_score': float(model_info['f1_score']),
            'precision': float(model_info['precision']),
            'total_features': len(feature_names),
            'features': feature_names
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/feature-names', methods=['GET'])
def get_features():
    """Get list of required feature names"""
    try:
        return jsonify({
            'features': feature_names,
            'total': len(feature_names)
        })
    except Exception as e:
        return jsonify({'error': f'Error getting features: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Verify model is loaded
        if model is None:
            return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_name': model_info['model_name'],
            'method': model_info['method_name']
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING FRAUD DETECTION WEB SERVER")
    print("="*70)
    print("\n📊 Available Endpoints:")
    print("  • GET  /                    - Main web interface")
    print("  • POST /predict             - Single transaction prediction")
    print("  • POST /predict-batch       - Multiple transactions prediction")
    print("  • GET  /model-info          - Get model information")
    print("  • GET  /feature-names       - Get required features")
    print("  • GET  /health              - Health check")
    print("\n🌐 Server starting at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True)