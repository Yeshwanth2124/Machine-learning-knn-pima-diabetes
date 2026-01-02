# KNN on Pima Indians Diabetes Dataset

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)

A machine learning project that implements the **K-Nearest Neighbors (KNN)** algorithm to predict the onset of diabetes based on diagnostic measures. This project utilizes the famous **Pima Indians Diabetes Database**.

## 📌 Project Overview

The goal of this project is to build a classification model that can accurately predict whether a patient has diabetes or not, based on certain health metrics. The project covers the entire machine learning pipeline, including:

1.  **Data Loading & Preprocessing**: Handling the dataset, checking for missing values (if any), and scaling features.
2.  **Exploratory Data Analysis (EDA)**: Understanding data distributions and correlations.
3.  **Model Training**: Implementing the KNN algorithm.
4.  **Evaluation**: Assessing model performance using accuracy and other metrics.

## 📂 Dataset Details

The dataset used is the **Pima Indians Diabetes Database**. It contains the following features for female patients at least 21 years old of Pima Indian heritage:

-   **Pregnancies**: Number of times pregnant
-   **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
-   **BloodPressure**: Diastolic blood pressure (mm Hg)
-   **SkinThickness**: Triceps skin fold thickness (mm)
-   **Insulin**: 2-Hour serum insulin (mu U/ml)
-   **BMI**: Body mass index (weight in kg/(height in m)^2)
-   **DiabetesPedigreeFunction**: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
-   **Age**: Age (years)
-   **Outcome**: Class variable (0 or 1), where 1 indicates the presence of diabetes.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Usage

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Yeshwanth2124/Machine-learning-knn-pima-diabetes.git
    cd Machine-learning-knn-pima-diabetes
    ```

2.  **Run the Notebook**

    Open the Jupyter Notebook `KNN_ON_PIMA_DATASET.ipynb` to view the analysis and run the code cells.

    ```bash
    jupyter notebook KNN_ON_PIMA_DATASET.ipynb
    ```

    > **Note**: The notebook may currently be configured to load data from Google Drive. If running locally, please update the file path in the code to point to the local `Dataset/diabetes.csv` file.

## 🧠 Algorithm: K-Nearest Neighbors (KNN)

KNN is a simple, supervised machine learning algorithm that can be used for both classification and regression tasks. It operates on the principle of "feature similarity":

1.  **Select K**: Choose the number of neighbors (k).
2.  **Calculate Distance**: Find the distance between the query instance and all training samples.
3.  **Sort**: Sort the distances and determine the k-nearest neighbors.
4.  **Vote**: For classification, return the mode (most frequent class) of the k-nearest neighbors.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or the analysis, feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information. (If applicable)

---
*Created by [Yeshwanth](https://github.com/Yeshwanth2124)*
