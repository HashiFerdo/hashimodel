# from keras.models import model_from_json

# Load the JSON file containing the model architecture
import json
import tensorflow as tf
from keras.models import load_model

# Load the InceptionV3 model without including top layers (i.e., fully connected layers)

# Load your saved model from model.h5 file
import numpy as np
from PIL import Image
import io



def process_image(image_data,m):
    
    # image = Image.open(image_data)
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image)
    # Divide by 255 to normalize pixel values
    processed_image = image_array

    if m == 1:
        model = load_model('my_model (2).keras')
        classes = ['algal leaf in tea',
 'bird eye spot in tea',
 'brown blight in tea',
 'healthy tea leaf',
 'red leaf spot in tea']
    elif m == 2:
        classes =['3 months - 1 year', 'Invalid Class', '1 year - 15 year']
        model = load_model('my_modelT6.keras')
    pred = model.predict(np.expand_dims(processed_image, axis=0))
    # print(pred)
    return classes[np.argmax(pred)]