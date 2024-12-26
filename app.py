from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np
import os


class DDoSModel(torch.nn.Module):
    def __init__(self):
        super(DDoSModel, self).__init__()
        self.fc1 = torch.nn.Linear(21, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.fc5 = torch.nn.Linear(16, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = torch.nn.LayerNorm(128)
        self.shortcut = torch.nn.Linear(21, 32)

    def forward(self, x):
        identity = x
        x = self.leaky_relu(self.layer_norm(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        identity = self.shortcut(identity)
        x += identity
        x = self.leaky_relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


app = Flask(__name__)

model = DDoSModel()
model.load_state_dict(torch.load('ddos_model.pth'))
model.eval()

scaler = joblib.load('scaler.pkl')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'dataFile' in request.files:
            file = request.files['dataFile']
            if file and file.filename.endswith('.txt'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    inputs = []
                    for line in lines:
                        line = line.strip()
                        line_data = line.split(',')
                        values = [float(value.split(':')[1]) for value in line_data]
                        inputs.append(values)

                    input_data = np.array(inputs)

                    input_scaled = scaler.transform(input_data)
                    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

                    with torch.no_grad():
                        predictions = model(input_tensor).round().numpy()

                    results = []
                    for prediction in predictions:
                        result = "DDoS attack detected." if prediction == 1 else "No DDoS attack detected."
                        results.append(result)

                    return render_template('result.html', results=results)
                except Exception as e:
                    return render_template('result.html', result=f"Error processing the file: {str(e)}")
            else:
                return "Invalid file format. Please upload a .txt file."
        else:
            return "No file provided."
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input, 'features' key is required."}), 400

        features = data["features"]

        if not isinstance(features, list) or len(features) != 21:
            return jsonify({"error": "Invalid format, 'features' must be a list of 21 numerical values."}), 400

        input_data = np.array(features).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).round().item()

        result = "DDoS attack detected" if prediction == 1 else "No DDoS attack detected"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True)
