from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image #to read images in python
#import tensorflow as tf
#from keras.layers import TFSMLayer  # Import TFSMLayer

app = FastAPI()

# MODEL = tf.keras.models.load_model('../saved_models/2')
#MODEL = TFSMLayer("../saved_models/2", call_endpoint="serving_default")
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

@app.get("/ping")
async def ping():
    return "Hello, I'm alive!"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)) #Read image as pillow image 
    image = np.array(image) #and convert to numpy array
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read() #file content as bytes
    image = read_file_as_image(bytes) #bytes to numpy array
    # img_batch = np.expand_dims(image,0)
    # prediction = MODEL.predict(img_batch)
    # console.log(type(prediction))
    return

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)