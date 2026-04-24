from contextlib import asynccontextmanager

import mlflow
import logging

import numpy as np
from scipy.special import softmax

from fastapi import FastAPI, HTTPException
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

mlflow.set_tracking_uri("http://mlflow:5000")

# model = mlflow.pyfunc.load_model("models:/iris_onnx_model/1")

# logger.info("Модель подгружена")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Этот код выполнится ОДИН РАЗ при старте сервера
    logger.info("Загрузка модели из MLflow...")
    try:
        model = mlflow.pyfunc.load_model("models:/iris_onnx_model/1")
        logger.info("Модель успешно подгружена")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель: {e}")
        # Можно даже остановить запуск, если модель критична
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=IrisPredictedClassResponse)
async def predict(req: IrisInputsRequest):
    logger.info("Получили запрос /predict")
    
    try:
        xs = np.array([req.x1, req.x2, req.x3, req.x4], dtype=np.float32)
        xs = xs.reshape(1, -1)
        logits = model.predict(xs)
        prediction = softmax(logits["output"])
    except Exception:
        logger.warning("Something wrong with model and inputs")

        raise HTTPException(
            status_code=400,
            detail="Not valid inputs"
        )
    predicted_class = int(np.argmax(prediction) + 1)

    logger.info("Модель обработала запрос")

    return {"iris_class": predicted_class}
