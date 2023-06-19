import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from st_files_connection import FilesConnection
from google.cloud import storage
import h5py
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./root-cortex-390013-fa63cf4ae748.json"

def darken_image(grayscale_img, constant, mask_threshold):
    mask = grayscale_img > mask_threshold
    grayscale_img[mask] -= constant
    return grayscale_img

def load_h5_model(bucket_name, source_blob_name):
    """Loads an HDF5 model from a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    
    local_model_filename = "./model.h5"
    blob.download_to_filename(local_model_filename)

    
    model = h5py.File(local_model_filename, "r")

    
    os.remove(local_model_filename)

    return model


bucket_name = "colorizationgrad"
source_blob_name = "model5unet_forest_500_light300e.h5"


h5_model = load_h5_model(bucket_name, source_blob_name)

def colorize_image(model, grayPic):
    grayPic = grayPic[:, :, ::-1].copy()
    grayPic = cv2.cvtColor(grayPic, cv2.COLOR_BGR2GRAY)
    
    grayPic = cv2.resize(grayPic, (128,128))
    grayPic = np.expand_dims(grayPic, axis=-1)
    grayPic = darken_image(grayPic, 40, 230)
    grayPic = np.broadcast_to(grayPic, (1, 128, 128, 3))
    prediction = model.predict([grayPic])
    generated_image = Image.fromarray((prediction[0]).astype('uint8')).resize((256, 256))
    generated_image = np.asarray(generated_image)

    
    return generated_image


def colorize_video(model, video):
   
    video_capture = cv2.VideoCapture(video)

   
    output_file = 'output_colorized.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_file, fourcc, 30.0, (256, 256))

   
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Colorize the frame
        colorized_frame = colorize_image(model, frame)

        # Write the colorized frame to the output video
        output_video.write(colorized_frame.astype(np.uint8))

    # Release video objects
    video_capture.release()
    output_video.release()

    return output_file


model = tf.keras.models.load_model(h5_model)


st.title('Colorization')


colorization_option = st.selectbox('Select colorization option', ('Image', 'Video'))

if colorization_option == 'Image':
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)

        
        if st.button('Colorize'):
           
            colorized_image = colorize_image(model, np.array(image))

            
            st.image(colorized_image, caption='Colorized Image', use_column_width=True)

elif colorization_option == 'Video':
    uploaded_file = st.file_uploader('Upload a video', type=['mp4'])

    if uploaded_file is not None:
        
        st.video(uploaded_file)

        
        if st.button('Colorize'):
            
            output_file = colorize_video(model, uploaded_file)

            
            st.video(output_file)
