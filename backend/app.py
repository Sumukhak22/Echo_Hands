from flask import Flask, request, jsonify
from flask_cors import CORS
from mongo_db import create_user, authenticate_user

app = Flask(__name__)
CORS(app)  

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    if create_user(username, email, password):
        return jsonify({"message": "User registered successfully"}), 201
    else:
        return jsonify({"error": "Username already exists"}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    is_valid = authenticate_user(username, password)

    if is_valid:
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

if __name__ == '__main__':
    app.run(debug=True)
