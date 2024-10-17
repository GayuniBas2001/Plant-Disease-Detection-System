import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from pydantic import BaseModel
from typing import List
from io import BytesIO
import io
from PIL import Image 

#from keras.layers import TFSMLayer  # Import TFSMLayer

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello, I'm alive!"

# MODEL = tf.keras.models.load_model('../saved_models/2')
# MODEL = TFSMLayer("../saved_models/2", call_endpoint="serving_default")
# model = keras.layers.TFSMLayer('../saved_models/my_model.h5', call_endpoint='serving_default')

model = keras.models.load_model('../saved_models/my_model.h5', compile=False)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=['accuracy'])
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

# Define a data model for the response
class Prediction(BaseModel):
    label: str
    confidence: float


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)) #Read image as pillow image 
    # image = image.resize((256, 256))  # Resizing to the input shape expected by your model
    image = np.array(image) #.astype('float32') / 255.0
    return image

# @app.post("/predict")
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read() #file content as bytes
    image = read_file_as_image(bytes) #bytes to numpy array
    img_batch = np.expand_dims(image,0)

    predictions = model.predict(img_batch)  
    print("predictions : ",predictions)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions[0])
    print("predicted_index : ",predicted_index)
    print("predicted_class : ",predicted_class)
    print("confidence : ",confidence)

    return {
        "class": predicted_class, 
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)











