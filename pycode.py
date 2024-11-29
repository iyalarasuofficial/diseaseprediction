import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming these are defined in symptoms_disease.py
from symptoms_disease import l1, disease


# Load and preprocess data
df = pd.read_csv("Training.csv")

df.replace({'prognosis': {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}}, inplace=True)

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Load test data
tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}}, inplace=True)

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

class DiseasePredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Disease Predictor")
        self.geometry("800x700")
        self.configure(bg="#f0f0f0")
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 12), padding=5)
        style.configure("TOptionMenu", font=("Arial", 12))

        # Header Frame
        header_frame = tk.Frame(self, bg="#4a4a4a")
        header_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            header_frame,
            text="Disease Predictor",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#4a4a4a"
        ).pack(pady=10)

        tk.Label(
            header_frame,
            text="Using Machine Learning",
            font=("Arial", 16),
            fg="white",
            bg="#4a4a4a"
        ).pack(pady=5)

        # Main Content Frame
        content_frame = tk.Frame(self, bg="#f0f0f0")
        content_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Input Frame
        input_frame = tk.Frame(content_frame, bg="#f0f0f0")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid to center the input_frame
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        self.name_var = tk.StringVar()
        self.symptom_vars = [tk.StringVar(value="None") for _ in range(5)]

        # Patient Name
        ttk.Label(input_frame, text="Patient Name:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        ttk.Entry(input_frame, textvariable=self.name_var, font=("Arial", 12)).grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        # Configure columns to expand
        input_frame.grid_columnconfigure(1, weight=1)

        # Symptom Selection
        for i in range(5):
            ttk.Label(input_frame, text=f"Symptom {i+1}:").grid(row=i+1, column=0, sticky="w", pady=5, padx=5)
            symptom_menu = ttk.OptionMenu(
                input_frame,
                self.symptom_vars[i],
                "None",
                *sorted(l1),
                command=lambda _: self.check_symptoms()
            )
            symptom_menu.grid(row=i+1, column=1, sticky="ew", pady=5, padx=5)

        # Button Frame
        button_frame = tk.Frame(content_frame, bg="#f0f0f0")
        button_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid to center the button_frame
        content_frame.grid_columnconfigure(1, weight=1)

        self.dt_button = ttk.Button(button_frame, text="Decision Tree", command=self.decision_tree, state="disabled")
        self.dt_button.pack(fill=tk.X, pady=5)

        self.rf_button = ttk.Button(button_frame, text="Random Forest", command=self.random_forest, state="disabled")
        self.rf_button.pack(fill=tk.X, pady=5)

        self.lr_button = ttk.Button(button_frame, text="Logistic Regression", command=self.logistic_regression, state="disabled")
        self.lr_button.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Reset", command=self.reset_fields).pack(fill=tk.X, pady=5)
        ttk.Label(button_frame, text="*Reset after every prediction", font=("Arial", 10), background="#f0f0f0").pack(fill=tk.X, pady=5)

        # Results Frame
        results_frame = tk.Frame(self, bg="#e0e0e0", bd=2, relief=tk.GROOVE)
        results_frame.pack(fill=tk.BOTH, padx=20, pady=10, expand=True)

        self.result_vars = [tk.StringVar() for _ in range(3)]
        result_labels = ["Decision Tree", "Random Forest", "Logistic Regression"]

        for i, label in enumerate(result_labels):
            tk.Label(results_frame, text=f"{label}:", font=("Arial", 12, "bold"), bg="#e0e0e0").grid(row=i, column=0, sticky="w", padx=10, pady=10)
            tk.Label(results_frame, textvariable=self.result_vars[i], font=("Arial", 12), bg="#e0e0e0").grid(row=i, column=1, sticky="w", padx=10, pady=10)

        # Centering the Results Frame
        for i in range(2):
            results_frame.grid_columnconfigure(i, weight=1)

    def check_symptoms(self, *args):
        if any(var.get() != "None" for var in self.symptom_vars):
            self.dt_button["state"] = "normal"
            self.rf_button["state"] = "normal"
            self.lr_button["state"] = "normal"
        else:
            self.dt_button["state"] = "disabled"
            self.rf_button["state"] = "disabled"
            self.lr_button["state"] = "disabled"

    def reset_fields(self):
        self.name_var.set("")
        for var in self.symptom_vars:
            var.set("None")
        for var in self.result_vars:
            var.set("")
        self.check_symptoms()

    def get_prediction(self, clf):
        symptoms = [var.get() for var in self.symptom_vars]
        # Reset l2 to all zeros before setting based on symptoms
        l2 = [1 if symptom in symptoms else 0 for symptom in l1]
        prediction = clf.predict([l2])[0]
        accuracy = accuracy_score(y_test, clf.predict(X_test)) * 100
        result = disease[prediction] if prediction < len(disease) else "Not Found"
        return f"{result} (Accuracy : {accuracy:.2f}%)"

    def decision_tree(self):
        clf = tree.DecisionTreeClassifier().fit(X, y)
        self.result_vars[0].set(self.get_prediction(clf))

    def random_forest(self):
        clf = RandomForestClassifier().fit(X, np.ravel(y))
        self.result_vars[1].set(self.get_prediction(clf))

    def logistic_regression(self):
        clf = LogisticRegression(max_iter=200).fit(X, np.ravel(y))
        self.result_vars[2].set(self.get_prediction(clf))

if __name__ == "__main__":
    app = DiseasePredictorApp()
    app.mainloop()


# # Disease Prediction Using Machine Learning

# ---

# ## Slide 1: Introduction

# - **Title**: Disease Predictor: Harnessing the Power of Machine Learning
# - **Subtitle**: Revolutionizing Medical Diagnosis
# - **Key Points**:
#   - Utilizes multiple ML algorithms
#   - User-friendly interface
#   - High accuracy predictions

# ![Medical AI concept](https://api.placeholder.com/400x200?text=Medical+AI+Concept)

# ---

# ## Slide 2: How It Works

# 1. **Input**: Patient symptoms (up to 5)
# 2. **Processing**: 
#    - Decision Tree
#    - Random Forest
#    - Logistic Regression
# 3. **Output**: Predicted disease with accuracy

# ```mermaid
# graph LR
#     A[Patient Symptoms] --> B[ML Models]
#     B --> C[Disease Prediction]
#     B --> D[Accuracy Score]
# ```

# ---

# ## Slide 3: Features and Benefits

# | Features | Benefits |
# |----------|----------|
# | Multiple ML algorithms | Increased prediction accuracy |
# | User-friendly GUI | Easy to use for medical staff |
# | Extensive symptom database | Covers wide range of diseases |
# | Real-time predictions | Quick decision support |

# ---

# ## Slide 4: Impact and Future

# - **Current Impact**:
#   - Assists in early disease detection
#   - Supports medical professionals in diagnosis
#   - Reduces time for initial assessment

# - **Future Enhancements**:
#   - Integration with electronic health records
#   - Continuous learning from new medical data
#   - Expansion to rare diseases prediction

# ![Future of Healthcare](https://api.placeholder.com/400x200?text=Future+of+Healthcare)