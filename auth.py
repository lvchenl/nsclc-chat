import csv
import os
from passlib.hash import bcrypt

CSV_FILE = "users.csv"

def init_csv():
    """Ensure the CSV file for user data exists and is initialized with headers."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])

def register_user(username, password):
    """
    Register a new user if the username doesn't already exist.
    Returns True if successful, False if user exists.
    """
    init_csv()
    username = username.strip()
    password = password.strip()
    if user_exists(username):
        return False
    hashed_pw = bcrypt.hash(password)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, hashed_pw])
    return True

def verify_user(username, password):
    """
    Verify if the username and password match an entry in the CSV file.
    """
    init_csv()
    username = username.strip()
    password = password.strip()
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username and bcrypt.verify(password, row["password"]):
                return True
    return False

def user_exists(username):
    """
    Check whether a user already exists in the CSV file.
    """
    init_csv()
    username = username.strip()
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        return any(row["username"] == username for row in reader)
