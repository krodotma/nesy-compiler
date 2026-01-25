from flask import Flask, request, jsonify
from .learner import MetaLearner

app = Flask(__name__)
learner = MetaLearner()

@app.route('/suggest', methods=['POST'])
def suggest():
    payload = request.json or {}
    # In a real implementation, we'd use the model to generate suggestions.
    # Here we return a placeholder design token suggestion.
    suggestion = {
        'design': 'glassmorphism_card',
        'tokens': {'--primary': 'hsl(210 50% 45%)', '--accent': 'hsl(30 70% 55%)'}
    }
    return jsonify(suggestion)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_path': learner.model.get('path') if isinstance(learner.model, dict) else 'none'})

if __name__ == '__main__':
    # Bind to all interfaces for VPS deployment
    app.run(host='0.0.0.0', port=8001)
