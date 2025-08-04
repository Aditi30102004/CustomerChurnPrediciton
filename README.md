Customer Churn Prediction System
This project is an end-to-end machine learning application designed to predict customer churn for a banking service. 
It features an interactive web interface built with Streamlit that allows users to input customer details and receive a real-time prediction on whether the customer is likely to leave the bank.
The application also includes a dashboard with visualizations to provide insights into the factors that influence customer churn.

🔴 Live Demo
The application is deployed on Streamlit Community Cloud and can be accessed here:
 **[View Live App](https://customerchurnprediciton-kmqkdj78rublknhreqeapt.streamlit.app/)**

App Screenshot

<img width="1917" height="872" alt="image" src="https://github.com/user-attachments/assets/74604d21-6148-4ef0-b2dc-26771ea3fcf6" />
<img width="1799" height="852" alt="image" src="https://github.com/user-attachments/assets/55b71662-a7f4-4140-962c-b18fdcd29de1" />

✨ Key Features
*   **Interactive UI:** A clean and user-friendly interface built with Streamlit, with input controls conveniently located in the sidebar.
*   **Real-time Prediction:** Get instant churn predictions and probabilities based on the input data.
*   **Data Visualization:** An integrated dashboard shows key insights, such as churn rates by geography and gender.
*   **Optimized Performance:** The machine learning model and data scaler are cached using `joblib`, ensuring fast startup times after the initial run.
*   **End-to-End Workflow:** Demonstrates the complete machine learning lifecycle from data exploration and preprocessing to model training and deployment.

🛠️ Tech Stack
*   **Language:** Python 3
*   **Libraries:**
    *   **Web Framework:** Streamlit
    *   **Data Manipulation:** Pandas, NumPy
    *   **Machine Learning:** Scikit-learn
    *   **Model Persistence:** Joblib

🚀 How to Run Locally
To run this application on your local machine, please follow the steps below.
### Prerequisites
*   Python 3.7+
*   Git

1. Clone the Repository
```sh
git clone https://github.com/Aditi30102004/CustomerChurnPrediciton.git
cd CustomerChurnPrediciton
```

2. Create a Virtual Environment (Recommended)
It's a good practice to create a virtual environment to manage project dependencies.
```sh
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
The required libraries are listed in the `requirements.txt` file.
```sh
pip install -r requirements.txt
```

4. Run the Streamlit App
Once the dependencies are installed, you can run the app with the following command:
```sh
streamlit run app.py
```

The application will open in your default web browser.

## 📂 Repository Structure

```
.
├── app.py                  # The main Streamlit application file
├── churn_model.joblib      # Saved machine learning model (generated on first run)
├── scaler.joblib           # Saved data scaler (generated on first run)
├── Churn_Modelling.csv     # The dataset used for training
├── model_train.py          # Script for model training and evaluation
├── explore_data.py         # Script for initial data exploration
└── requirements.txt        # List of Python dependencies
```

## License
This project is licensed under the MIT License.
