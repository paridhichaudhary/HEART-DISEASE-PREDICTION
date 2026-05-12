import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')

def main():
    # Load the objects from the single file in the EXACT same order
    try:
        with open('heart_model.pkl', 'rb') as f:
            model = pickle.load(f)
            scaler = pickle.load(f)
            features = pickle.load(f)
    except FileNotFoundError:
        print("Error: heart_model.pkl not found!")
        return

    print("\n--- Heart Disease Predictor ---")
    try:
        # User inputs
        age = float(input("Age: "))
        trestbps = float(input("Resting Blood Pressure: "))
        chol = float(input("Cholesterol: "))
        thalach = float(input("Max Heart Rate (thalach): "))
        oldpeak = float(input("ST Depression (oldpeak): "))
        ca = float(input("Major Vessels (0-3): "))
        thal = float(input("Thal (3, 6, or 7): "))
        cp = int(input("Chest Pain Type (1-4): "))

        # Prepare the data (Matching your notebook features)
        data = {
            'thalach': [thalach], 'chol': [chol], 'age': [age],
            'trestbps': [trestbps], 'oldpeak': [oldpeak],
            'ca': [ca], 'thal': [thal], 'cp_4': [1 if cp == 4 else 0]
        }
        df = pd.DataFrame(data)

        # Scale continuous columns
        cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
        df[cont_cols] = scaler.transform(df[cont_cols])
        
        # Match feature order and Predict
        df = df[features]
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        print(f"\nResult: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
        print(f"Confidence: {prob*100:.2f}%\n")

    except ValueError:
        print("Invalid input. Please enter numbers only.")

if __name__ == "__main__":
    main()