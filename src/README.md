Refund Item Classification System (Spotlight: Batch Processing)

Course: DLBDSMTP01 – Project: From Model to Production

Institution: IU International University of Applied Sciences

Project Overview
This repository contains an end-to-end Machine Learning pipeline designed for an e-commerce sustainable fashion platform. The system automates the categorization of customer returned items into 10 distinct categories, significantly reducing manual sorting workforce.

Key Components

Model Training (src/train.py): Trains a lightweight, production-ready Random Forest classifier on Fashion-MNIST images, achieving ~85% baseline accuracy without expensive GPU overhead. Generates confusion matrix and serializes the model artifact (models/refund_classifier.joblib).

Nightly Batch Pipeline (src/batch_runner.py): Scheduled batch processor (simulated via Cronjob @ 02:00 AM) that ingests all pending images from data/raw_incoming/, executes vectorized batch inference, generates a timestamped ERP-compatible CSV report with warehouse aisle destinations, and archives processed items to data/processed/.

RESTful API Service (src/api.py): FastAPI application providing on-demand HTTP POST endpoints (/predict-batch) with Pydantic schema validation for real-time integration.

System Architecture
See output/architecture_diagram.png for the complete 5-stage data flow:
Data Ingestion -> Raw Storage (Inbox) -> Nightly Batch Pipeline -> Business Output (CSV) -> Archive Storage.

Quickstart & Reproduction
1. Setup Environment:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Train the Model:
python src/train.py
3. Run Nightly Batch Processing:
python src/batch_runner.py
4. Launch RESTful API:
uvicorn src.api:app --reload
(Interactive API docs available at http://127.0.0.1:8000/docs)
