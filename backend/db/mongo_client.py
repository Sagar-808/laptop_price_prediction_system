from pymongo import MongoClient
from datetime import datetime
import os

class MongoDBClient:
    def __init__(self, uri: str | None = None, db_name: str = 'laptop_price_db', collection: str = 'predictions'):
        """Initialize MongoDB connection for laptop price app."""
        try:
            # Connect to MongoDB (simple local default, can be overridden with URI)
            if uri:
                self.client = MongoClient(uri)
            else:
                self.client = MongoClient('localhost', 27017)
            # Enforce default DB name unless overridden via param or env
            env_db = os.getenv('LAPTOP_PRICE_DB_NAME')
            effective_db = env_db or db_name or 'laptop_price_db'
            self.db = self.client[effective_db]
            self.collection = self.db[collection]
            print("MongoDB connected successfully")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def insert_prediction(self, specs: dict, predicted_price: float):
        """Insert laptop specs and predicted price into MongoDB.
        Args:
            specs (dict): Laptop specification fields submitted by user
            predicted_price (float): Model's predicted price
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.collection is None:
                print(" MongoDB not connected")
                return False
            
            # Create document to insert
            document = {
                'timestamp': datetime.now(),
                'specs': specs,
                'predicted_price': predicted_price
            }
            
            # Insert into database
            result = self.collection.insert_one(document)
            print(f"Prediction stored in MongoDB with ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            print(f"Failed to store prediction: {e}")
            return False

    def save_prediction(self, prediction_data: dict):
        """Save arbitrary prediction record for laptop price app."""
        try:
            if self.collection is None:
                print("MongoDB not connected")
                return False
            
            # Ensure timestamp is stored as datetime for consistent sorting
            ts = prediction_data.get('timestamp')
            if isinstance(ts, str):
                try:
                    prediction_data['timestamp'] = datetime.fromisoformat(ts)
                except Exception:
                    prediction_data['timestamp'] = datetime.now()
            elif not isinstance(ts, datetime):
                prediction_data['timestamp'] = datetime.now()

            # Insert into database
            result = self.collection.insert_one(prediction_data)
            print(f"Prediction stored in MongoDB with ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            print(f"Failed to store prediction: {e}")
            return False

    def get_predictions(self, limit=100):
        """Retrieve recent predictions from database."""
        try:
            if self.collection is None:
                print("MongoDB not connected")
                return []
            
            predictions = list(self.collection.find().sort('timestamp', -1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                if '_id' in pred:
                    pred['_id'] = str(pred['_id'])
                if 'timestamp' in pred:
                    ts = pred['timestamp']
                    if isinstance(ts, datetime):
                        pred['timestamp'] = ts.isoformat()
                    else:
                        # Already a string or other type; convert to str to be safe
                        pred['timestamp'] = str(ts)
            
            return predictions
            
        except Exception as e:
            print(f"Failed to retrieve predictions: {e}")
            return []

    def get_all_predictions(self, limit=100):
        """Alias for get_predictions (legacy)."""
        return self.get_predictions(limit)

    def get_prediction_stats(self):
        """Get basic statistics about stored predictions."""
        try:
            if self.collection is None:
                return {}
            
            total_predictions = self.collection.count_documents({})
            with_price = self.collection.count_documents({'predicted_price': {'$exists': True}})
            return {
                'total_predictions': total_predictions,
                'with_price': with_price,
            }
            
        except Exception as e:
            print(f"Failed to get prediction stats: {e}")
            return {}

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

"""Global client instance (not used by app routes but available for utilities)."""
mongo_client = MongoDBClient()

# Create convenience functions for direct import
def save_prediction(prediction_data: dict):
    """Convenience function to save a prediction record."""
    return mongo_client.save_prediction(prediction_data)

def get_predictions(limit=100):
    """Convenience function to get predictions."""
    return mongo_client.get_predictions(limit)