# rps-classification

A Rock Paper Scissors image classification project built with TensorFlow, FastAPI, and Streamlit.

## Tech Stack

- TensorFlow — model training and inference
- FastAPI — REST API for predictions
- Streamlit — interactive web dashboard
- Pillow & NumPy — image preprocessing

## Project Structure

```
rps-classification/
├── app/
│   └── main.py        # FastAPI application
├── models/
│   └── rps_baseline_model.keras
├── dashboard.py       # Streamlit dashboard
└── requirements.txt
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
# Start the FastAPI server
uvicorn app.main:app --reload

# Start the Streamlit dashboard
streamlit run dashboard.py
```