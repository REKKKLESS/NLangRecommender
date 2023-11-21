from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the language detection model
with open('model.pckl', 'rb') as model_file:
    Lrdetect_Model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_language', methods=['POST'])
def detect_language():
    input_test = request.form['input_test']
    result = Lrdetect_Model.predict([input_test])[0]
    return jsonify(result=result)

@app.route('/home.html')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
