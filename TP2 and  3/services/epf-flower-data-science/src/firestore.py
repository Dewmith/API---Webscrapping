from google.cloud import firestore
from google.oauth2 import service_account

class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client with explicit credentials."""
        credentials = service_account.Credentials.from_service_account_file(
            "S:/OneDrive - Fondation EPF/Uni/5eme Année/Data Sources/API/API---Webscrapping/datasourceapi-54c3a-8fb030c2c3c6.json"
        )
        self.client = firestore.Client(credentials=credentials)

    def get(self, collection_name: str, document_id: str) -> dict:
        """Find one document by ID."""
        doc = self.client.collection(collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileExistsError(
            f"No document found at {collection_name} with the id {document_id}"
        )

    def set(self, collection_name: str, document_id: str, data: dict) -> None:
        """Create or update a document."""
        try:
            self.client.collection(collection_name).document(document_id).set(data)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create or update document in {collection_name}/{document_id}: {e}"
            )
