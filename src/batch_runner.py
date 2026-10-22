#Importing all needed libraries
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

#Making sure each item of clothing has index number
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Assigning names to the paths
MODEL_PATH = "models/refund_classifier.joblib"
INCOMING_DIR = "data/raw_incoming"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
print("--- [NIGHTLY BATCH PROCESS STARTED] ---")
print(f"Loading trained model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

#Checking for unprocessed refund images
image_files = [f for f in os.listdir(INCOMING_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} unprocessed refund images in '{INCOMING_DIR}'.")

#Informing user if there are pictures to process
if not image_files:
    print("No new images to process tonight. Shutting down batch job gracefully.")
    exit(0)
print("Converting raw image files into model-ready features...")
#adjusting shape and size and preparing batch
batch_features = []
for file_name in image_files:
    file_path = os.path.join(INCOMING_DIR, file_name)
    
    img = Image.open(file_path).convert('L').resize((28, 28))
    
    pixels = np.array(img).reshape(28 * 28)
    batch_features.append(pixels)

X_batch = np.array(batch_features)

#Running batch prediction
print(f"Running batch prediction on {len(image_files)} items...")
predictions = model.predict(X_batch)
probabilities = model.predict_proba(X_batch)
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

#Separating recognized pictures by aisles
for i, file_name in enumerate(image_files):
    pred_class = CLASS_NAMES[predictions[i]]
    confidence = np.max(probabilities[i]) * 100
    
    if any(shoe in pred_class.lower() for shoe in ["boot", "sandal", "sneaker"]):
        aisle = "Aisle_B_Footwear"
    elif "bag" in pred_class.lower():
        aisle = "Aisle_C_Accessories"
    else:
        aisle = "Aisle_A_Clothing"

    results.append({
        "image_file": file_name,
        "predicted_category": pred_class,
        "confidence": f"{confidence:.1f}%",
        "warehouse_destination": aisle,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

#Preparing and saving report
report_df = pd.DataFrame(results)
today_date = datetime.now().strftime("%Y-%m-%d")
report_path = os.path.join(OUTPUT_DIR, f"refund_decisions_{today_date}.csv")
report_df.to_csv(report_path, index=False)
print(f"Business report saved to: {report_path}")

## Archiving processed images to keep the incoming folder clean for the next day
os.makedirs(PROCESSED_DIR, exist_ok=True)
for file_name in image_files:
    src_path = os.path.join(INCOMING_DIR, file_name)
    dst_path = os.path.join(PROCESSED_DIR, file_name)
    shutil.move(src_path, dst_path)

print(f"Archived {len(image_files)} processed files to '{PROCESSED_DIR}'.")
print("--- [NIGHTLY BATCH PROCESS COMPLETED SUCCESSFULLY] ---")