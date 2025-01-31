from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse , JSONResponse,HTMLResponse
from pydantic import BaseModel

import uvicorn
import os
import warnings
import tensorflow as tf

from src.custom_logger import setup_logger
from src.inference import inference


app=FastAPI(title="Portuguese into English Translate",
    description="FastAPI",
    version="0.115.4")

# Allow all origins (replace * with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

logger=setup_logger("app",api_app=not None)
warnings.filterwarnings("ignore")



@app.get("/")
async def root():
  return {"Fast API":"API is working"}


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out info, 2 = filter out warnings, 3 = filter out errors
warnings.filterwarnings("ignore")


@app.post("Portuguese2English")    
async def Portuguese_English_Translate(portu_sentence:str):
  
    result=inference(portu_sentence)
    if result["status"]==1:
        return {"Status":result["status"], "English_Translate":result["English_Translate"]}
    else:
        logger.error(f"Exception error: {result['Massage']}",exc_info=True)
        return {"Status":result["status"],"Message":result["Massage"]}




if __name__=="__main__":

    logger.info(f"Num GPUs Available: , {len(tf.config.list_physical_devices('GPU'))}")
    
    uvicorn.run("app:app",host="0.0.0.0", port=8000, reload=True)
