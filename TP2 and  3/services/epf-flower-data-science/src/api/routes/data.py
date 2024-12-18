from fastapi import APIRouter
import opendatasets as od
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import numpy as np
from fastapi import HTTPException

router = APIRouter()

# Step 6: Endpoint to download the Iris dataset
@router.get("/download-dataset")
async def download_dataset():
    """
    Downloads the Iris dataset from Kaggle and saves only the CSV file to the src/data folder.
    """
    try:
        # Define the dataset URL and the destination folder
        dataset_url = "https://www.kaggle.com/datasets/uciml/iris"
        destination = "TP2 and  3/services/epf-flower-data-science/src/data"

        # Download the dataset
        od.download(dataset_url, data_dir=destination)

        return {"message": f"CSV file(s) downloaded and moved to {destination}"}

    except Exception as e:
        return {"error": str(e)}

# Step 7: Endpoint to load the Iris dataset and return it as a DataFrame in JSON format
@router.get("/load-dataset")
async def load_dataset():
    """
    Loads the Iris dataset from the src/data folder, converts it to a DataFrame, and returns it as JSON.
    """
    try:
        # Path to the downloaded CSV file
        dataset_path = "TP2 and  3/services/epf-flower-data-science/src/data/iris/Iris.csv"

        # Check if the dataset file exists
        if not os.path.exists(dataset_path):
            return {"error": f"Dataset not found at {dataset_path}"}

        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Convert the DataFrame to a dictionary and return it as JSON
        data_json = df.to_dict(orient="records")

        return {"data": data_json}

    except Exception as e:
        return {"error": str(e)}

# Step 8: Processing the dataset
@router.get("/process-dataset")
async def process_dataset():
    """
    Process the Iris dataset by handling missing values, encoding categorical data, and scaling features.
    """
    try:
        # Path to the dataset
        dataset_path = "TP2 and  3/services/epf-flower-data-science/src/data/iris/Iris.csv"

        # Path to save the preprocessed dataset
        preprocessed_path = "TP2 and  3/services/epf-flower-data-science/src/data/preprocessed.csv"


        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        #Coverting the 'species' column to string
        df['Species'] = df['Species'].str.split().str[0]

        # Encode categorical 'species' column
        label_encoder = LabelEncoder()
        df['Species'] = label_encoder.fit_transform(df['Species'])
    
        # Handle missing values (if any)
        df.fillna(df.mean(), inplace=True)  # Fills missing numerical values with the mean

        # Save the preprocessed DataFrame to CSV
        df.to_csv(preprocessed_path, index=False)

        # Convert the DataFrame to a dictionary and return it as JSON
        data_json = df.to_dict(orient="records")

        return {"data": data_json}

    except Exception as e:
        return {"error": str(e)}
    
# Step 9: Split dataset into train and test
@router.get("/split-dataset")
async def split_dataset():
    """
    Splits the Iris dataset into training and testing sets and returns them as JSON.
    """
    try:
        # Path to the dataset
        dataset_path = "TP2 and  3/services/epf-flower-data-science/src/data/preprocessed.csv"

        # Folder to save the train and test splits
        save_dir = "TP2 and  3/services/epf-flower-data-science/src/data"

        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Split into features and target
        X = df.drop(columns=['Species',"Id"])  # Features (sepal_length, sepal_width, etc.)
        y = df['Species']  # Target (species)

        # Split the dataset into train and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the train and test splits to dictionaries for DataFrames (X_train, X_test)
        X_train_json = X_train.to_dict(orient="records")
        X_test_json = X_test.to_dict(orient="records")

        # Convert the Series (y_train, y_test) to lists
        y_train_json = y_train.tolist()
        y_test_json = y_test.tolist()

        #Converting the data set to a json file to be saved, to be used in the next step to easily extract the data.
        data_splits = {
            "X_train": X_train_json,
            "X_test": X_test_json,
            "y_train": y_train_json,
            "y_test": y_test_json
        }


        # Save all splits into a single JSON file
        save_path = os.path.join(save_dir, "train_test_splits.json")
        with open(save_path, "w") as f:
            json.dump(data_splits, f)

        return data_splits

    except Exception as e:
        return {"error": str(e)}


# Step 11: Train a classification model
@router.post("/train-model")
async def train_model():
    """
    Trains a classification model using the processed train-test split data and saves the model in the models folder.
    """
    try:
        # Paths to the files
        splits_path = "TP2 and  3/services/epf-flower-data-science/src/data/train_test_splits.json"
        params_path = "TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json"
        model_save_path = "TP2 and  3/services/epf-flower-data-science/src/models"
        
        # Ensure the model save directory exists
        os.makedirs(model_save_path, exist_ok=True)

        # Load the train-test splits
        if not os.path.exists(splits_path):
            return {"error": f"Train-test splits not found at {splits_path}"}
        
        with open(splits_path, "r") as f:
            data_splits = json.load(f)
        
        # Extract train-test data
        X_train = pd.DataFrame(data_splits["X_train"])
        X_test = pd.DataFrame(data_splits["X_test"])
        
        y_train = pd.Series(data_splits["y_train"])
        y_test = pd.Series(data_splits["y_test"])

        # Load model parameters
        if not os.path.exists(params_path):
            return {"error": f"Model parameters not found at {params_path}"}
        
        with open(params_path, "r") as f:
            model_params = json.load(f)

        # Initialize and train the classification model (Random Forest in this case)
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Evaluate the model
        #y_pred = model.predict(X_test)
        #accuracy = accuracy_score(y_test, y_pred)

        # Save the trained model
        model_file_path = os.path.join(model_save_path, "classification_model.joblib")
        joblib.dump(model, model_file_path)

        print(f"Model type before saving: {type(model)}")
        print(f"Model details: {model}")


        return {
            "message": "Model trained and saved successfully.",
            "model_path": model_file_path,
        }
    
    except Exception as e:
        return {"error": str(e)}

    

# Step 12: Prediction with Trained Model
@router.post("/predict")
async def predict(features: dict):
    """
    Predicts the species of iris flowers based on the trained model and input features.
    Parameters:
        features (dict): A dictionary where keys are feature names (e.g., "sepal_length", "sepal_width", etc.) and values are the feature values.
    Returns:
        dict: Predicted class labels as JSON.
    """
    try:
        # Path to the trained model
        model_path = "TP2 and  3/services/epf-flower-data-science/src/models/classification_model.joblib"

        # Check if the model exists
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Trained model not found. Please train the model first.")

        # Load the trained model
        model = joblib.load(model_path)
        
        # Print the model type to check if it's correctly loaded
        print(f"Model type after loading: {type(model)}")

        # Validate input features
        required_features = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
        for feature in required_features:
            if feature not in features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required feature: {feature}. Required features are: {', '.join(required_features)}"
                )

        # Prepare input data as a numpy array
        input_data = np.array([[features[feature] for feature in required_features]])

        # Make predictions
        predictions = model.predict(input_data)

        if predictions.tolist()[0] == 0:
            return {"prediction": "Iris-setosa"}
        elif predictions.tolist()[0] == 1:
            return {"prediction": "Iris-versicolor"}
        elif predictions.tolist()[0] == 2:
            return {"prediction": "Iris-virginica"}

    except Exception as e:
        return {"error": str(e)}