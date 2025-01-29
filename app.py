from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd

# Load GPT-Neo for conversational model
chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Load the CSV file
df = pd.read_csv('/sample_health_recommendations_1000 (1).csv')

# Initialize the FastAPI app
app = FastAPI()

# Pydantic models for request bodies
class HealthData(BaseModel):
    glucose_level: float
    recent_intake: str
    weight: float
    family_history: str
    health_tip_input: str

# Endpoint to handle the conversation and provide recommendations
@app.post("/chat/")
async def chat_conversation(data: HealthData):
    glucose_level = data.glucose_level
    recent_intake = data.recent_intake.lower()
    weight = data.weight
    family_history = data.family_history.lower()
    health_tip_input = data.health_tip_input.lower()

    # Conversation flow based on glucose level
    if recent_intake == 'yes':
        return {"message": "It seems your glucose level may have spiked due to recent intake. Please try again later with an empty stomach."}
    
    # Provide basic glucose recommendations
    if glucose_level < 70:
        recommendation = "Your glucose is low. You may want to eat a small snack with some carbs."
    elif 70 <= glucose_level <= 100:
        recommendation = "Your glucose level is normal. Keep maintaining a healthy diet and regular exercise."
    elif 100 < glucose_level <= 130:
        recommendation = "Your glucose level is a bit high. Consider reducing your sugar intake and getting some exercise."
    else:
        recommendation = "Your glucose level is quite high. It may be a good idea to consult a healthcare provider."
    
    # Use GPT-Neo to generate a more conversational recommendation response
    chat_input = f"Based on the glucose level of {glucose_level} mg/dL, what health recommendation would you give?"
    generated_response = chatbot(chat_input, max_length=100, num_return_sequences=1, truncation=True)[0]['generated_text']

    # Print the chatbot's recommendation
    message = generated_response.strip() + "\n" + recommendation

    # Ask about family history of diabetes and diagnose based on that
    if family_history == 'yes':
        message += "\nYou might be at risk for Type 2 diabetes. Regular monitoring and a healthy lifestyle are key."
    else:
        message += "\nSince there is no family history, you might have Type 1 or gestational diabetes. Please consult a doctor for further advice."
    
    # Offer health tips from the CSV dataset
    if health_tip_input == 'yes':
        health_tips = df['Health Tip'].sample(1).values[0]
        message += f"\nHere is a health tip: {health_tips}"

    # End the conversation with a friendly message
    chat_input = "Give me a friendly closing statement to end this health conversation."
    closing_message = chatbot(chat_input, max_length=50, num_return_sequences=1, truncation=True)[0]['generated_text']
    message += f"\n{closing_message.strip()}"

    return {"message": message}

# Run the application using Uvicorn (in terminal)
# uvicorn filename:app --reload
