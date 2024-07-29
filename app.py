from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load the model and encoders
try:
    model = joblib.load('model.pkl')
    activity_encoder = joblib.load('activity_encoder.pkl')
    health_tip_encoder = joblib.load('health_tip_encoder.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading models or encoders: {e}")

app = FastAPI()

# Define a request model
class GlucoseLevelRequest(BaseModel):
    glucose_level: float

# Define a response model
class RecommendationResponse(BaseModel):
    activity_suggestion: str
    health_tip: str

@app.post('/recommend', response_model=RecommendationResponse)
def recommend(request: GlucoseLevelRequest):
    try:
        # Get the glucose level from the request
        glucose_level = request.glucose_level
        
        # Make a prediction
        prediction = model.predict(np.array([[glucose_level]]))
        
        # Ensure prediction has the expected shape
        if prediction.shape[1] != 2:
            raise ValueError("Prediction output has an unexpected shape.")
        
        # Decode the predictions
        activity_suggestion = activity_encoder.inverse_transform([int(prediction[0, 0])])[0]
        health_tip = health_tip_encoder.inverse_transform([int(prediction[0, 1])])[0]
        
        # Create the response
        response = RecommendationResponse(
            activity_suggestion=activity_suggestion,
            health_tip=health_tip
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with uvicorn if this script is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
