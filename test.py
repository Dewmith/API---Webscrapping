from google.cloud import firestore
from google.oauth2 import service_account

def test_firestore_connection():
    # Explicitly specify the credentials file
    credentials = service_account.Credentials.from_service_account_file(
        'S:/OneDrive - Fondation EPF/Uni/5eme Année/Data Sources/API/API---Webscrapping/datasourceapi-54c3a-8fb030c2c3c6.json'
    )

    # Initialize Firestore client with the specified credentials
    db = firestore.Client(credentials=credentials)
    print("OUTPUT:",db)

    try:
        # Reference to the 'parameters' collection and 'parameters' document
        doc_ref = db.collection("parameters").document("parameters")

        # Get the document
        doc = doc_ref.get()

        if doc.exists:
            print("Document data:", doc.to_dict())
        else:
            print("No such document! Creating document...")
            # Create the document with some initial data
            doc_ref.set({
                "n_estimators": 100,
                "criterion": "gini"
            })
            print("Document created with default values.")

    except Exception as e:
        print(f"Error occurred: {e}")

test_firestore_connection()
