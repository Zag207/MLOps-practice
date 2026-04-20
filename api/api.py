import mlflow
import logging

import numpy as np
from scipy.special import softmax

from fastapi import FastAPI
from pydantic import BaseModel


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

class IrisInputsRequest(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float

class IrisPredictedClassResponse(BaseModel):
    iris_class: int

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model("models:/iris_classification/1")

logger.info("Модель подгружена")

app = FastAPI()

@app.post("/predict", response_model=IrisPredictedClassResponse)
async def predict(req: IrisInputsRequest):
    logger.info("Получили запрос /predict")
    
    xs = np.array([req.x1, req.x2, req.x3, req.x4], dtype=np.float32)
    logits = model.predict(xs)
    prediction = softmax(logits)
    predicted_class = int(np.argmax(prediction) + 1)

    logger.info("Модель обработала запрос")

    return {"iris_class": predicted_class}
