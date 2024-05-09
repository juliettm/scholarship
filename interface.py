import gradio as gr
import numpy as np
import joblib
from sklearn.tree import _tree
import pandas as pd


# Sample model function for loan prediction
# Load the Decision Tree model
model_path = 'decision_tree_scholarship_model.joblib'
tree_model = joblib.load(model_path)
# Get the decision path for the input
feature_names = ['Gender', 'EA', 'AL', 'PC', 'PT', 'AE', 'EComplete', 'IncorrectDuration', 'SIncomplete', 'Less60', 'Program']
   # ['Gender', 'Academic Record', 'Language proficiency', 'Scientific Production', 'Work Plan Quality', 'Suitability of the destination institution', 'Enrollment Complete', 'Destiny and Duration Correctness', 'Application Complete', 'Less60', 'Program']
# 'Gender', 'EA', 'AL', 'PC', 'PT', 'AE', 'EComplete', 'IncorrectDuration', 'SIncomplete', 'Less60', 'Program'

def get_decision_path_details(tree_model, single_instance, feature_names):
    decision_path = tree_model.decision_path(single_instance).indices
    decision_features = []
    decision_thresholds = []
    decision_directions = []

    tree_ = tree_model.tree_

    for node_index in decision_path:
        if tree_.feature[node_index] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree_.feature[node_index]]
            threshold = tree_.threshold[node_index]
            decision = single_instance.iloc[0, tree_.feature[node_index]] <= threshold
            direction = "<=" if decision else ">"

            decision_features.append(feature_name)
            decision_thresholds.append(threshold)
            decision_directions.append(f"{feature_name} {direction} {threshold:.2f}")

    decisions = " -> ".join(decision_directions)
    return decisions
def loan_prediction(Gender, EA, AL, PC, PT, AE, EComplete, IncorrectDuration, SIncomplete, Program):

    # Model
    model = tree_model
    # Sample input transformation
    program = {"CIENCIAS DE LA SALUD":0, "CIENCIAS, TECNOLOGÍAS E INGENIERÍAS":1, "HUMANIDADES, CIENCIAS SOCIALES Y JURÍDICAS":2}
    gender = {"Male": 1, "Female": 0}
    Less60 = 1 if EA + AL + PC + PT + AE < 60 else 0
    EC = 1 if EComplete == "Yes" else 0
    ID = 1 if IncorrectDuration == "Yes" else 0
    SI = 0 if SIncomplete == "Yes" else 1

    input_data = np.array([[gender[Gender],
                            EA,
                            AL,
                            PC,
                            PT,
                            AE,
                            EC,
                            ID,
                            SI,
                            Less60,
                            program[Program]
                            ]])
    # Sample prediction
    prediction = model.predict(input_data)
    # Sample output, replace this with your actual output
    output = "Approved" if prediction == 1 else "Rejected"

    # Prepare data for prediction
    input_df = pd.DataFrame([input_data[0]],
                            columns=['Gender', 'EA', 'AL', 'PC', 'PT', 'AE', 'EComplete', 'IncorrectDuration', 'SIncomplete', 'Less60', 'Program'])


    decision_path_text = get_decision_path_details(tree_model, input_df, feature_names)

    results = "The client is classified as: " + output + ". The decision path is: " + decision_path_text

    return (output, decision_path_text)

# Define inputs for the Gradio interface
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Academic Record", minimum=0, maximum=30),
    gr.Number(label="Language proficiency", minimum=0, maximum=5),
    gr.Number(label="Scientific Production", minimum=0, maximum=25),
    gr.Number(label="Work Plan Quality", minimum=0, maximum=20),
    gr.Number(label="Suitability of the destination institution", minimum=0, maximum=20),
    gr.Radio(label="Enrollment Complete", choices=["Yes", "No"]),
    gr.Radio(label="Destiny and Duration Correctness", choices=["Yes", "No"]),
    gr.Radio(label="Application Complete", choices=["Yes", "No"]),
    gr.Dropdown(
        ["CIENCIAS DE LA SALUD", "CIENCIAS, TECNOLOGÍAS E INGENIERÍAS", "HUMANIDADES, CIENCIAS SOCIALES Y JURÍDICAS"],
        label="Program")
]


# Define the output component for the Gradio interface
output = [gr.Textbox(label="Scholarship application"), gr.Textbox(label="Decision Path")]


# Create the Gradio interface
gr.Interface(fn=loan_prediction, inputs=inputs, outputs=output, title="Loan Prediction App").launch(share=True)

