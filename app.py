from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("C:/Users/Harsha/New folder/flightpredection-main/flight_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature list
with open("C:/Users/Harsha/New folder/flightpredection-main/model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        if request.method == "POST":
            # Parse form data
            dep_time = request.form["Dep_Time"]
            journey_day = pd.to_datetime(dep_time).day
            journey_month = pd.to_datetime(dep_time).month
            dep_hour = pd.to_datetime(dep_time).hour
            dep_min = pd.to_datetime(dep_time).minute

            arrival_time = request.form["Arrival_Time"]
            arrival_hour = pd.to_datetime(arrival_time).hour
            arrival_min = pd.to_datetime(arrival_time).minute

            dur_hour = abs(arrival_hour - dep_hour)
            dur_min = abs(arrival_min - dep_min)

            total_stops = int(request.form["stops"])

            airline = request.form["airline"]
            source = request.form["Source"]
            destination = request.form["Destination"]

            # Prepare input dict for model
            input_data = {
                "Total_Stops": total_stops,
                "Journey_day": journey_day,
                "Journey_month": journey_month,
                "Dep_hour": dep_hour,
                "Dep_min": dep_min,
                "Arrival_hour": arrival_hour,
                "Arrival_min": arrival_min,
                "Dur_hour": dur_hour,
                "Dur_min": dur_min,
            }

            # One-hot encode categorical features
            for col in model_features:
                if col.startswith("Airline_"):
                    input_data[col] = 1 if col == f"Airline_{airline}" else 0
                elif col.startswith("Source_"):
                    input_data[col] = 1 if col == f"Source_{source}" else 0
                elif col.startswith("Destination_"):
                    input_data[col] = 1 if col == f"Destination_{destination}" else 0

            # Fill missing features with 0
            for col in model_features:
                if col not in input_data:
                    input_data[col] = 0

            # Create DataFrame with correct feature order
            final_input = pd.DataFrame([input_data])[model_features]

            # Predict price
            prediction = model.predict(final_input)
            output = round(prediction[0], 2)

            return render_template("home.html", predictions=f"You will have to Pay approx Rs. {output}")

    except Exception as e:
        return render_template("home.html", predictions=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
