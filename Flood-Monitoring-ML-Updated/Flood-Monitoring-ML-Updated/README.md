# Flood Monitoring and Prediction System Using Machine Learning

## 1. Project Overview

The **Flood Monitoring and Prediction System** is a machine learning-based web application developed using Python and Streamlit. It allows users to enter environmental feature values and obtain a model-generated `FloodProbability` score.

The application also supports batch predictions through CSV file uploads, displays feature contributions to help explain individual predictions, and warns users when input combinations fall outside the range observed during model training.

**Important:** The model's output is a dataset-based estimate, not a verified real-world flood probability. The current application does not use live weather data, river sensors, or real-time flood alerts.

## 2. Objectives

- Develop a machine learning application for flood-related score prediction.
- Provide an interactive interface for individual predictions.
- Support batch predictions using CSV files.
- Visualize feature contributions to individual predictions.
- Identify inputs outside the training data range.
- Provide downloadable prediction results.

## 3. Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas and NumPy
- **Visualization:** Streamlit charts and plotting components
- **Testing:** Pytest

## 4. Key Features

### Individual Prediction
Users can enter feature values and generate a `FloodProbability` score using the trained machine learning model.

### Feature Contribution Visualization
A graph displays the contributions of input features to the model's prediction, helping users understand how the model arrived at its output.

### Batch Prediction
Users can upload a CSV file containing multiple records, generate predictions for valid rows, review the results, and download the output CSV.

### Extrapolation Warning
The application warns when the sum of the input features is outside the training range of 63–144. Such inputs may cause the model to extrapolate beyond the data it learned from.

### Bounded Score Display
For individual predictions, the displayed score is restricted to the range of 1%–100%. This is a presentation rule only; the underlying model output and downloaded CSV values are not changed by this display cap.

## 5. Project Structure

The following is a representative structure. Confirm it against your actual project folder before publishing.

```text
Flood-Monitoring-ML-Updated/
├── app.py
├── src/
│   └── formatting.py
├── tests/
│   ├── test_dashboard.py
│   └── test_percentage_display.py
├── .venv/
└── README.md
```

Other files, including the dataset, model artifacts, and dependency configuration, may also be present in the project.

## 6. Installation and Setup

### Prerequisites

- Python installed on your computer
- pip package manager
- Visual Studio Code or another code editor

### Step 1: Open the project folder

Open the project folder in Visual Studio Code and open a terminal.

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is restricted by your PowerShell policy, you can run the project's Python executable directly as shown below.

### Step 4: Install dependencies

Install the dependencies listed in the project's dependency file, if one exists. For example:

```powershell
python -m pip install -r requirements.txt
```

If the project does not contain `requirements.txt`, create it from the actual packages and versions used by your application before publishing these instructions.

## 7. Run the Application

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the application.

## 8. Run the Tests

Execute the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The verified test run completed with **71 passed, 9 skipped, 0 failed, and 23 subtests passed**.

## 9. Limitations

- The model depends on the dataset and training process used in this project.
- The displayed score is not necessarily a calibrated probability of an actual flood.
- The 1%–100% display cap does not improve the model's predictive accuracy.
- Inputs outside the training range may produce unreliable extrapolated predictions.
- The current application does not collect live weather, rainfall, river-level, or sensor readings.
- The current application does not provide verified real-time flood warnings.

## 10. Future Enhancements

- Integrate live weather and rainfall data from a reliable API.
- Add river-level and sensor data where available.
- Evaluate alternative machine learning models.
- Validate predictions against historical flood events.
- Add a map-based flood monitoring interface.
- Develop a real-time alert system after suitable data sources and validation are available.
- Improve model evaluation and probability calibration.

## 11. Disclaimer

This project is intended for educational and demonstration purposes. Its predictions should not be used as the sole basis for emergency decisions, evacuation planning, or public safety actions. Consult official weather departments and disaster management authorities for actual flood warnings.

## 12. Author

**SUGALI DILEEP NAIK**

B.Tech — Computer Science and Engineering  
VIT-AP University
