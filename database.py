
import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("Please set MONGO_URI in Render.")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10000
)

db = client["TrafficViolationDetector"]
