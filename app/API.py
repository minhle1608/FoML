from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import call_func

app = FastAPI()

class DataInput(BaseModel):
    brand: str
    seller_type: int
    transmission: int
    year: int
    owner: int
    fuel_type: int
    engine: float
    max_power: float

@app.post('/predict')
async def predict_price(data: DataInput):
    try:
        res = call_func(
            brand=data.brand,
            seller_type=data.seller_type,
            transmission=data.transmission,
            year=data.year,
            owner=data.owner,
            fuel_type=data.fuel_type,
            engine=data.engine,
            max_power=data.max_power
        )
        return {"prediction": float(res) }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))