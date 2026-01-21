from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load("model/house_price_model.pkl")

# Features for the form
numerical_features = [
    "OverallQual", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "BedroomAbvGr", "FullBath", "YearBuilt"
]

neighborhoods = [
    "BrDale","BrkSide","ClearCr","CollgCr","Crawfor",
    "Edwards","Gilbert","NAmes","NPkVill","OldTown",
    "SWISU","Sawyer","SawyerW","Somerst","StoneBr",
    "Timber","Veenker"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get numerical inputs
        data = {feature: float(request.form[feature]) for feature in numerical_features}
        # Get categorical input
        data["Neighborhood"] = request.form["Neighborhood"]
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        # Make prediction
        prediction = round(pipeline.predict(input_df)[0], 2)
    return render_template("index.html", prediction=prediction,
                           numerical_features=numerical_features,
                           neighborhoods=neighborhoods)

if __name__ == "__main__":
    app.run(debug=True)
