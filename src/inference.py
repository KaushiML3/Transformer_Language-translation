
import tensorflow as tf
import os
from src.custom_logger import setup_logger

logger=setup_logger("inference")

#load model
current_direction = os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join("artifact/model/version_1")
model = tf.saved_model.load("artifact/model/version_1")




def inference(sentence):
    try:
        predict=model(sentence).numpy()

    except Exception as e:
        logger.error(f"Exception error: {str(e)}",exc_info=True)
        return{"status":0,"Message":str(e)}

    else:
        return {"status":1,"English_Translate":predict}
    
