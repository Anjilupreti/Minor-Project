from flask import Flask, request, jsonify,render_template,send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
import pickle
import cv2
import base64
import os

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

# Load model
model = tf.keras.models.load_model('fashion.h5')

# Load fashion item data
fashion_data = pd.read_pickle('filenames.pkl')
#item_features = fashion_data(['filename','type'])
item_ids = fashion_data['type_label'].values

# Embeddings
with open('embeddings(1).pkl', 'rb') as f:
    output = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    image_filename = None
    if request.method == 'POST' and 'image' in request.files:
        # Save the uploaded image to the upload folder
        image = request.files['image']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD'], filename))
        image_filename = filename
    return render_template('index.html', image_filename=image_filename)

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD'], filename)

@app.route('/recommend',methods = ['POST'])
def upload_file():
    image = request.files['image']
    img = Image.open(image)
    img_gray= img.convert('L')
    resize_img = img_gray.resize((28,28))
    x_data = np.array(resize_img).reshape(-1,28,28,1)
    x_data=x_data/255

    # Make prediction
    prediction = model.predict(x_data)
    item_id = np.argmax(prediction)
    print(x_data)
    print(item_id)

    # Get recommended items
    neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='cosine')
    neighbors.fit(output)
    distances,indices = neighbors.kneighbors(prediction)
    recommended_items = [fashion_data.iloc[index] for index in indices[0][1:]]
    print(recommended_items)
    response = [{'filename': item['filename'], 'type': item['type']} for item in recommended_items]
    

        # Load recommended images
    recommended_images = []
    for item in recommended_items:
        filename = '/home/typhoon/Desktop/minor project/images/'+item['filename']
        img = cv2.imread(filename)
        recommended_images.append(img)

    # Return recommended images and metadata
    response = []
    for item, img in zip(recommended_items, recommended_images):
        filename = item['filename']
        image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        response.append({
            'filename': filename,
            'type': item['type'],
            'image': base64.b64encode(image_bytes).decode('utf-8')
        })

    return jsonify(response)


if(__name__ == "__main__"):
    app.run()   
