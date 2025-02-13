# Breast Cancer Prediction

This repository contains a data science project focused on predicting breast cancer diagnoses using machine learning algorithms. By leveraging various classification models, the project aims to analyze medical data and provide a reliable prediction system for distinguishing between malignant and benign diagnoses. This is a practical example of how data science can contribute to healthcare by improving diagnostic accuracy and assisting medical professionals.

## Features

- **Data Preprocessing**: Handles missing values and encodes categorical data.
- **Data Visualization**: Includes histograms, pairplots, and correlation heatmaps.
- **Model Implementation**: Trained and tested multiple machine learning models.
- **Model Evaluation**: Provides accuracy scores, confusion matrices, and classification reports.
- **Hyperparameter Tuning**: Utilizes GridSearchCV for optimal parameter selection.
- **Cross-Validation**: Implements k-fold cross-validation for robust performance evaluation.

## Models Used

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)

## Tools and Libraries

- Python
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn, Plotly (visualization)
- Scikit-learn (machine learning)
- Pickle (model serialization)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`data.csv`) in the project directory.

## Usage

1. Run the script to preprocess the data and train models:
   ```bash
   python breast_cancer_prediction.py
   ```

2. View accuracy scores and confusion matrices to evaluate models.

3. Hyperparameter tuning results are displayed for Decision Tree, KNN, and SVC models.

4. Use the serialized Logistic Regression model (`logistic_model.pkl`) for predictions.

## Dataset

The dataset should be a CSV file with columns such as `radius_mean`, `texture_mean`, `perimeter_mean`, and `diagnosis`. Ensure the dataset is preloaded at `/content/data.csv`.

## Results

- The highest accuracy model achieved **XX%** (replace with your actual results).
- Logistic Regression model is saved as `logistic_model.pkl` for deployment.

## Future Work

- Expand feature engineering to include more predictors.
- Deploy the model using Flask or FastAPI for real-time predictions.
- Explore deep learning models for improved accuracy.

## Contribution

Feel free to fork this repository, create issues, or submit pull requests to enhance the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
