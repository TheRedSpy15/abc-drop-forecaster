import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# Load past event dates from the text file
with open('past_dates.txt', 'r') as file:
    previous_dates = [line.strip() for line in file]

# Add the current date to the list of previous dates
current_date = datetime.now().strftime("%Y-%m-%d")
previous_dates.append(current_date)

# Convert dates to numerical values (days since the first date)
base_date = datetime.strptime(previous_dates[0], "%Y-%m-%d")
numeric_dates = [(datetime.strptime(date, "%Y-%m-%d") - base_date).days for date in previous_dates]

# Create a linear regression model
model = LinearRegression()
X = np.array(numeric_dates).reshape(-1, 1)
y = np.arange(len(previous_dates)).reshape(-1, 1)
model.fit(y, X)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['GET'])
def predict_next_date():
    next_index = len(previous_dates)
    predicted_days_since_base = model.predict(np.array([[next_index]])).flatten()[0]
    predicted_date = base_date + timedelta(days=int(predicted_days_since_base))
    
    # Ensure predicted date is not in the past
    if predicted_date < datetime.now():
        predicted_date = datetime.now() + timedelta(days=1)
    
    response = {
        "predicted_next_date": predicted_date.strftime("%Y-%m-%d")
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()