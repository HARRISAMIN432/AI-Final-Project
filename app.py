from flask import Flask
from flask_cors import CORS
import os
from routes import register_routes
from utils import setup_logging

app = Flask(__name__)
CORS(app)

setup_logging()

register_routes(app)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("="*80)
    app.run(debug=True, host='0.0.0.0', port=5000)