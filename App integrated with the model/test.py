import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import pyrebase
from firebase_admin import db
import requests
import os
from firebase_functions import storage_fn
import time
from google.cloud import storage
from subprocess import Popen, PIPE, STDOUT
import sys
import subprocess
from post_training import main

firebaseConfig = {
  "apiKey": "AIzaSyCRwfK8ago4z2w_L5rAtuxk7fVRsLpPOPs",
  "authDomain": "miller-s-ale-house-final.firebaseapp.com",
  "databaseURL": "https://miller-s-ale-house-final-default-rtdb.firebaseio.com",
  "projectId": "miller-s-ale-house-final",
  "storageBucket": "miller-s-ale-house-final.appspot.com",
  "messagingSenderId": "506800754058",
  "appId": "1:506800754058:web:0b071acc3e3785e8250c73",
  "measurementId": "G-88M1BC8F2X"
};

cred = credentials.Certificate('./key.json')  # Replace with your service account key path
app = firebase_admin.initialize_app(cred, name='[DEFAULT]', options={
    'databaseURL': firebaseConfig['databaseURL'],
    'storageBucket': firebaseConfig['storageBucket']
})

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
list_images_done=[]
# text=[]

def get_image_from_storage_input():
    # get the image_name through json listeners
    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    # find difference
    data = get_json_from_database()
    input_imgs = data['input_images']



    for input_image_key, input_image_value in input_imgs.items():
        print(f'input_images/{input_image_value}')
        download_path = f'images_to_process/{input_image_value}'
        storage.child(f'input_images/{input_image_value}').download("", download_path)
        # download_path_with_newline = download_path + '\n'
        # break
    
        # p = Popen(['python3', 'post_training.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        # stdout_data = p.communicate(input=download_path_with_newline.encode("utf-8"))[0]

        # # print("here")

        if  download_path not in list_images_done:

            text=main(download_path)

            # text = subprocess.run(['python', 'post_training.py', download_path])

            # text = text.stdout
            # print(text)
            list_images_done.append(download_path)
            post_image_to_storage_output(text_in=str(text), filename=input_image_value )
        # with open('downloaded_image_input_paths.txt', 'a') as file:
        #     file.write(download_path + '\n')
        #     print("Download path appended to file")

def post_image_to_storage_output(text_in, filename):
    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()
    
    database_reference = db.reference('json/output_images')

    # for filename in os.listdir('./output'):
    print("ytss")
    path_cloud = f"output_images/{filename}"
    path_local = f"./output/{filename}"
    img_path_json = f"output_images/{filename}"
    storage.child(path_cloud).put(path_local)

    print("Image uploaded to storage output")
    data = get_json_from_database()

    filename_without_extension = filename.split('.')[0]

    output_images = data.get('output_images', {})

    # if 'output_images' not in data:
    #     data['output_images'] = {}
    

    # data['output_images'] = {
    #     filename_without_extension: {
    #         'image_name': filename,
    #         'similarity_score': 90
    #     }
    # }
    output_images[filename_without_extension] = {
        'image_name': filename,
        'similarity_score': text_in
    }
    
    # Update the data dictionary with the modified output_images
    data['output_images'] = output_images

    print(data)

    post_json_to_database(data)
        # data = get_json_from_database()

def get_json_from_database():
    ref = db.reference('/json')
    
    # Get JSON data from Firebase Realtime Database
    data = ref.get()
    print("JSON data retrieved from database")
    # print(data)
    return data

def post_json_to_database(data):
    ref = db.reference('/json')
    
    # Post JSON data to Firebase Realtime Database
    ref.update(data)
    print("JSON data posted to database")


# JSON data to be posted
img_path_json = ""



# Call the functions
# post_image_to_storage_input()
get_image_from_storage_input()
# post_image_to_storage_output()
# post_image_to_storage_output()
data = get_json_from_database()
# get_image_from_storage_output()



def handle_change(event):
    get_image_from_storage_input()

ref = db.reference('/json')
listener = ref.listen(handle_change)


     




