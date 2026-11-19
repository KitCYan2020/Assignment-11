from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from xgboost import XGBRegressor
import pandas as pd

# Load the pre-trained model
# model_path = os.path.join(os.path.abspath(__file__), "model", "model.json")
# model = XGBRegressor()
# model.load_model(model_path)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "monthly-car-sales.csv")
model_dir = os.path.join(base_dir, "model")
model_path = os.path.join(model_dir, "model.json")

# Load the dataset
df = pd.read_csv(data_path)

# Create lag features (sales from the last 6 months) from 1month - 6 months
for i in range(1, 7):
    df[f'lag_{i}'] = df['Sales'].shift(i)

# Drop rows with NaN values resulting from shift operation
df_features = df.dropna().reset_index(drop=True)

# Features are the 6 lag month columns; target are the current Sales
X = df_features[[f'lag_{i}' for i in range(1, 7)]]
y = df_features['Sales']

# Use XGBRegressor to train the model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X, y)


model.save_model(model_path)


# Initialize FastAPI app
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class SalesInput(BaseModel):
    sales: list[float]  # Expect exactly 6 values


# Mount the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Define the prediction endpoint
@app.post("/predict")
async def predict(input: SalesInput):
    if len(input.sales) != 6:
        return {"error": "Exactly 6 months of sales data required."}
    
    data = np.array(input.sales).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
