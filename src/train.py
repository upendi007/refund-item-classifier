#Importing all needed libraries
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Making sure each item of clothing has index number
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Loading data, making sure during testing model is not training anymore
train_data = datasets.FashionMNIST(root="./data", train=True, download=True)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True)

#Adjusting data for scikit-learning requirements by convering squere to line of numbers.
X_train = train_data.data[:10000].numpy().reshape(-1, 28 * 28)
y_train = train_data.targets[:10000].numpy()

X_test = test_data.data[:2000].numpy().reshape(-1, 28 * 28)
y_test = test_data.targets[:2000].numpy()

print(f"Data ready! Training samples: {X_train.shape[0]}, Features per image: {X_train.shape[1]}")

#Implementing Random Forest Algorithm 
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("Model training complete!")

predictions = model.predict(X_test)

#Creating Classification Report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions, target_names=CLASS_NAMES))

#Creating space fo our reports or giving an option to use existing one
os.makedirs("output", exist_ok=True)

#Checking model predictions with reality
cm = confusion_matrix(y_test, predictions)

#Visualizing and formatting confusion matrix
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Refund Department: Classification Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")

print("Confusion Matrix saved to: output/confusion_matrix.png")

#Creating space for our model or giving an option to use existing one
os.makedirs("models", exist_ok=True)

#Saving our training
joblib.dump(model, "models/refund_classifier.joblib")
print("Model successfully saved to: models/refund_classifier.joblib")

# Create an 'incoming' storage directory to hold unprocessed refund images from customers
raw_incoming_dir = "data/raw_incoming"
os.makedirs(raw_incoming_dir, exist_ok=True)

# Extract 10 sample images, sanitize category names, and save them as PNG files to simulate incoming customer returns
for i in range(10):
    img, label = test_data[i]
    clean_label_name = CLASS_NAMES[label].replace('/', '_')
    img.save(os.path.join(raw_incoming_dir, f"refund_item_{1000 + i}_{clean_label_name}.png"))

print(f"Sample incoming refund images saved to '{raw_incoming_dir}'.")