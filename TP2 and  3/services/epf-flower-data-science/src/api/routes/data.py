from fastapi import APIRouter
import opendatasets as od
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

        #Coverting the 'specied' column to string
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


        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Split into features and target
        X = df.drop(columns=['Species'])  # Features (sepal_length, sepal_width, etc.)
        y = df['Species']  # Target (species)

        # Split the dataset into train and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the train and test splits to dictionaries for DataFrames (X_train, X_test)
        X_train_json = X_train.to_dict(orient="records")
        X_test_json = X_test.to_dict(orient="records")

        # Convert the Series (y_train, y_test) to lists
        y_train_json = y_train.tolist()
        y_test_json = y_test.tolist()

        return {
            "X_train": X_train_json,
            "X_test": X_test_json,
            "y_train": y_train_json,
            "y_test": y_test_json
        }

    except Exception as e:
        return {"error": str(e)}
