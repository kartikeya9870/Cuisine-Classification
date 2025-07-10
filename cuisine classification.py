import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import os
import webbrowser
import numpy as np

df = pd.read_csv("restaurant_data.xlsx.csv")
df.columns = df.columns.str.strip()
df.dropna(subset=['Cuisines'], inplace=True)

features = ['Country Code', 'City', 'Longitude', 'Latitude', 'Price range',
            'Aggregate rating', 'Votes', 'Has Table booking', 'Has Online delivery']
target = 'Cuisines'

le_cuisine = LabelEncoder()
df[target] = le_cuisine.fit_transform(df[target])

label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

df.dropna(subset=features, inplace=True)

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

unique_labels = np.unique(y_test)
target_names = le_cuisine.inverse_transform(unique_labels)
report_text = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)

with open("classification_report.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

root = tk.Tk()
root.title("Cuisine Classifier")
root.geometry("560x780")
root.configure(bg="#f0f0f5")

tk.Label(root, text="Enter Restaurant Details", font=("Arial", 18, "bold"), bg="#f0f0f5", fg="#333").pack(pady=10)

entries = {}
for col in features:
    frame = tk.Frame(root, bg="#f0f0f5")
    frame.pack(pady=5)
    tk.Label(frame, text=col + ":", font=("Arial", 12), width=22, anchor="w", bg="#f0f0f5").pack(side="left")
    entry = ttk.Entry(frame, width=25)
    entry.pack(side="left")
    entries[col] = entry

def predict_cuisine():
    try:
        input_data = []
        record = {}
        for col in features:
            val = entries[col].get().strip()
            record[col] = val
            if col in label_encoders:
                val = val.title()
                val = label_encoders[col].transform([val])[0]
            else:
                val = float(val)
            input_data.append(val)

        pred = model.predict([input_data])[0]
        cuisine_name = le_cuisine.inverse_transform([pred])[0]
        record['Predicted Cuisine'] = cuisine_name
        
        history_file = "prediction_history.xlsx"
        if os.path.exists(history_file):
            existing_df = pd.read_excel(history_file)
            new_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
        else:
            new_df = pd.DataFrame([record])
        new_df.to_excel(history_file, index=False)

        messagebox.showinfo("Prediction", f"Predicted Cuisine: {cuisine_name}")
        show_result_graph(record)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input:\n{e}")

def show_result_graph(record):
    try:
        labels = list(record.keys())[:-1]  
        values = []
        for k in labels:
            try:
                values.append(float(record[k]))
            except:
                values.append(0)

        plt.figure(figsize=(8, 5))
        bars = plt.barh(labels, values, color='skyblue')
        plt.title(f"Predicted Cuisine: {record['Predicted Cuisine']}")
        plt.xlabel("Value")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Graph Error:", e)

def show_eval():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def view_history():
    if os.path.exists("prediction_history.xlsx"):
        webbrowser.open("prediction_history.xlsx")
    else:
        messagebox.showinfo("No History", "No prediction history found yet.")

ttk.Button(root, text="Predict Cuisine", command=predict_cuisine).pack(pady=15)
tk.Label(root, text=f"Model Accuracy: {round(acc*100, 2)}%", font=("Arial", 12, "italic"), bg="#f0f0f5", fg="green").pack()

ttk.Button(root, text="Show Evaluation Metrics", command=show_eval).pack(pady=10)
ttk.Button(root, text="View Prediction History", command=view_history).pack(pady=10)
ttk.Button(root, text="Open Classification Report", command=lambda: webbrowser.open("classification_report.txt")).pack(pady=10)

root.mainloop()
