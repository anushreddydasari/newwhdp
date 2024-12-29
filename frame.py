import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

class HeartDiseasePredictorApp:
    def __init__(self, root, model_path, scaler_path):
        self.root = root
        self.root.title("Heart Disease Prediction")

        # Load the trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Create labels and entry widgets for user input
        self.create_widgets()

    def create_widgets(self):
        # Labels and Entry fields for user input
        self.labels = {
            "Age": tk.Label(self.root, text="Age:"),
            "Sex": tk.Label(self.root, text="Sex (1 for Male, 0 for Female):"),
            "Chest Pain Type": tk.Label(self.root, text="Chest Pain Type (0-3):"),
            "Resting BP": tk.Label(self.root, text="Resting Blood Pressure (mm Hg):"),
            "Cholesterol": tk.Label(self.root, text="Cholesterol (mg/dl):"),
            "Fasting Blood Sugar": tk.Label(self.root, text="Fasting Blood Sugar (1 if > 120 mg/dl, else 0):"),
            "Resting ECG": tk.Label(self.root, text="Resting ECG (0-2):"),
            "Max Heart Rate": tk.Label(self.root, text="Max Heart Rate (bpm):"),
            "Exercise Induced Angina": tk.Label(self.root, text="Exercise Induced Angina (1 for Yes, 0 for No):"),
            "Oldpeak": tk.Label(self.root, text="Oldpeak (Depression in ST segment):"),
            "Slope": tk.Label(self.root, text="Slope (0-2):"),
            "CA": tk.Label(self.root, text="Number of Major Vessels (0-3):"),
            "Thal": tk.Label(self.root, text="Thalassemia (3 for Normal, 6 or 7 for Abnormal):")
        }

        # Input fields (entry widgets)
        self.entries = {
            key: tk.Entry(self.root) for key in self.labels.keys()
        }

        # Place labels and entry fields on the window
        for i, (key, label) in enumerate(self.labels.items()):
            label.grid(row=i, column=0, padx=10, pady=5)
            self.entries[key].grid(row=i, column=1, padx=10, pady=5)

        # Submit button to make the prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.labels), column=0, columnspan=2, pady=10)

    def get_user_input(self):
        """
        Collects the user input and returns it as a pandas DataFrame.
        """
        try:
            # Collect data from entries
            data = {
                "age": float(self.entries["Age"].get()),
                "sex": int(self.entries["Sex"].get()),
                "cp": int(self.entries["Chest Pain Type"].get()),
                "trestbps": float(self.entries["Resting BP"].get()),
                "chol": float(self.entries["Cholesterol"].get()),
                "fbs": int(self.entries["Fasting Blood Sugar"].get()),
                "restecg": int(self.entries["Resting ECG"].get()),
                "thalach": float(self.entries["Max Heart Rate"].get()),
                "exang": int(self.entries["Exercise Induced Angina"].get()),
                "oldpeak": float(self.entries["Oldpeak"].get()),
                "slope": int(self.entries["Slope"].get()),
                "ca": int(self.entries["CA"].get()),
                "thal": int(self.entries["Thal"].get())
            }

            # Convert to DataFrame for prediction
            return pd.DataFrame([data])

        except ValueError:
            # Handle invalid input (e.g., non-numeric values)
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return None

    def preprocess_data(self, new_data):
        """
        Scales the user input data using the pre-fitted scaler.
        """
        return self.scaler.transform(new_data)

    def predict(self):
        """
        Makes the prediction based on user input and displays the result.
        """
        new_data = self.get_user_input()

        if new_data is not None:
            # Preprocess the input data (scaling)
            scaled_data = self.preprocess_data(new_data)

            # Make prediction
            prediction = self.model.predict(scaled_data)[0]

            # Display the result
            if prediction == 1:
                messagebox.showinfo("Prediction", "The model predicts: Heart Disease (1)")
            else:
                messagebox.showinfo("Prediction", "The model predicts: No Heart Disease (0)")

# Main function to run the app
def main():
    # Load paths to your model and scaler
    model_path = 'random_forest_model.pkl'
    scaler_path = 'scaler.pkl'

    # Create the Tkinter root window
    root = tk.Tk()

    # Create the HeartDiseasePredictorApp instance
    app = HeartDiseasePredictorApp(root, model_path, scaler_path)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
