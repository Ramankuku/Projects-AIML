from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3

app = FastAPI(title="SageMaker Prediction API")

class InputData(BaseModel):
    Tenure: float
    City_Tier: float
    Service_Score: float
    Account_user_count: float
    account_segment: float
    Complain_ly: float
    Day_Since_CC_connect: float
    cashback: float

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
ENDPOINT_NAME = "rf-prediciton-endpoint-new"

@app.post("/predict")
def predict(data: InputData):
    try:
        payload_list = [
            data.Tenure,
            data.City_Tier,
            data.Service_Score,
            data.Account_user_count,
            data.account_segment,
            data.Complain_ly,
            data.Day_Since_CC_connect,
            data.cashback
        ]

        payload = ",".join(map(str, payload_list))

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",  
            Body=payload
        )

        result = response['Body'].read().decode('utf-8')

        return {"prediction": result}

    except runtime.exceptions.ModelError as me:
        raise HTTPException(status_code=500, detail=f"SageMaker ModelError: {me}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend Error: {e}")
