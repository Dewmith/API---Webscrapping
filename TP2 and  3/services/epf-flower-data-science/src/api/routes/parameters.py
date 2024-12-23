from fastapi import APIRouter
from src.firestore import FirestoreClient

router = APIRouter()

@router.post("/create-parameters")
async def create_parameters(n_estimators: int, criterion: str):
    """
    Create or update Firestore document with parameters.
    """
    try:
        # Initialize Firestore client
        firestore_client = FirestoreClient()

        # Define collection and document name
        collection_name = "parameters"
        document_id = "parameters"

        # Data to save
        data = {
            "n_estimators": n_estimators,
            "criterion": criterion,
        }

        # Save the parameters to Firestore
        firestore_client.set(collection_name, document_id, data)

        return {"message": "Parameters saved successfully.", "data": data}

    except Exception as e:
        return {"error": str(e)}
    

@router.get("/retrieve-parameters")
async def retrieve_parameters():
    """
    Retrieve parameters from Firestore.
    """
    try:
        # Initialize Firestore client
        firestore_client = FirestoreClient()

        # Retrieve the parameters
        data = firestore_client.get("parameters", "parameters")

        return {"message": "Parameters retrieved successfully.", "data": data}

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/update-parameters")
async def update_parameters(n_estimators: int = None, criterion: str = None):
    """
    Update or add parameters in Firestore.
    """
    try:
        # Initialize Firestore client
        firestore_client = FirestoreClient()

        # Retrieve existing parameters to avoid overwriting other fields
        collection_name = "parameters"
        document_id = "parameters"
        existing_data = firestore_client.get(collection_name, document_id)

        # Update fields only if provided
        if n_estimators is not None:
            existing_data["n_estimators"] = n_estimators
        if criterion is not None:
            existing_data["criterion"] = criterion

        # Save the updated parameters
        firestore_client.set(collection_name, document_id, existing_data)

        return {"message": "Parameters updated successfully.", "data": existing_data}

    except FileNotFoundError:
        # Create new document if not found
        data = {}
        if n_estimators is not None:
            data["n_estimators"] = n_estimators
        if criterion is not None:
            data["criterion"] = criterion

        firestore_client.set("parameters", "parameters", data)
        return {"message": "Parameters added successfully.", "data": data}

    except Exception as e:
        return {"error": str(e)}

