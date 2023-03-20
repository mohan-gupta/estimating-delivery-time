from contextlib import asynccontextmanager

from fastapi import FastAPI

from schemas import Data
from predict import load_models, get_preds

model={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model["model"] = load_models()
    yield
    # Clean up the ML models and release the resources
    model.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(data: Data):
    pred = get_preds(data.dict(), model["model"])
    
    return {"prediction": pred}