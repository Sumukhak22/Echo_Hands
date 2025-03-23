from pymongo import MongoClient

# Connecting to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Login"]
users_collection = db["users"]

def create_user(username, email, password):
    """Save a new user to the database if they don't already exist."""
    if users_collection.find_one({"username": username}):
        return False  # Username already exists
    users_collection.insert_one({"username": username, "email": email, "password": password})
    return True

def authenticate_user(username, password):
    """Check if the provided username and password exist in the database."""
    user = users_collection.find_one({"username": username, "password": password})
    return user is not None
