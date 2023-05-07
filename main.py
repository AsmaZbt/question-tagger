from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import predict_pipeline


app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    tags:  list

    class Config:
        orm_mode = True



@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=PredictionOut, status_code=200)
def predict(payload: TextIn):
    tags = predict_pipeline(payload.text)
    if not tags:
        raise HTTPException(status_code=400, detail="Model not found.")
    
    return {"tags": tags}
