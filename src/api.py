#Importing all needed libraries
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

#Initializing FastAPI application with metadata for documentation
app = FastAPI(
    title="Refund Item Classifier API",
    description="RESTful API for automated batch and single item clothing classification in refunds department",
    version="1.0"
)

#Making sure each item of clothing has index number
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Assigning the name ot the path
MODEL_PATH = "models/refund_classifier.joblib"
model = joblib.load(MODEL_PATH)

#Setting expected format for batch images (list of pixels)
class BatchImageRequest(BaseModel):
    images: List[List[float]]

#Define home function
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Refund Item Classification API",
        "model_loaded": True
    }

#Define predict_batch
@app.post("/predict-batch")
def predict_batch(payload: BatchImageRequest):
    try:
        #Converting incoming images to numpy array
        input_data = np.array(payload.images)
                
        #Checking if each image has correct size of 784 pixels
        if input_data.shape[1] != 784:
            raise HTTPException(status_code=400, detail="Each image must have exactly 784 pixels (28x28).")
        
        #Running prediction and getting probabilities
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        #Preparing response with item name and confidence score
        response = []
        for i in range(len(predictions)):
            predicted_class = CLASS_NAMES[predictions[i]]
            confidence = float(np.max(probabilities[i]))
            
            response.append({
                "item_index": i,
                "predicted_category": predicted_class,
                "confidence_score": round(confidence, 4),
                "is_confident": confidence > 0.60
            })
            
        return {"total_items_processed": len(response), "results": response}
   
    #Handling errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))