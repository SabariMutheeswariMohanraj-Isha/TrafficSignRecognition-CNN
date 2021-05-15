import tkinter as tk
import tkinter.filedialog as fd
from tkinter import *
from PIL import Image, ImageTk
from tkmacosx import Button
from tkinter.font import Font

import numpy as np
import tensorflow
from keras.models import load_model

model = load_model("traffic_sign_classifier.h5")

classes = { 1:'Speed limit (20km/h)',2:'Speed limit (30km/h)',3:'Speed limit (50km/h)',4:'Speed limit (60km/h)',5:'Speed limit (70km/h)',6:'Speed limit (80km/h)',7:'End of speed limit (80km/h)',8:'Speed limit (100km/h)',9:'Speed limit (120km/h)',10:'No passing',11:'No passing veh over 3.5 tons',12:'Right-of-way at intersection',13:'Priority road',14:'Yield',15:'Stop',16:'No vehicles',17:'Veh > 3.5 tons prohibited',18:'No entry',19:'General caution',20:'Dangerous curve left',21:'Dangerous curve right', 22:'Double curve',23:'Bumpy road',24:'Slippery road',25:'Road narrows on the right',26:'Road work',27:'Traffic signals',28:'Pedestrians',29:'Children crossing',30:'Bicycles crossing',31:'Beware of ice/snow',32:'Wild animals crossing',33:'End speed + passing limits',34:'Turn right ahead',35:'Turn left ahead',36:'Ahead only',37:'Go straight or right',38:'Go straight or left',39:'Keep right',40:'Keep left',41:'Roundabout mandatory',42:'End of no passing',43:'End no passing veh > 3.5 tons' }

top = tk.Tk()
top.geometry("1100x600")
top.title("Traffic Sign Recognizer")
top.configure(background="#c7dcff")

label = tk.Label(top, background = "#c7dcff", font = ("montserrat",15,"bold"))
sign_image = tk.Label(top)



def classify(filepath):
    
    global label_packed
    
    img = Image.open(filepath)
    img = img.resize((30,30))
    img = np.expand_dims(img,axis=0)
    img = np.array(img)
    
    pred = model.predict_classes([img])[0]
    sign = classes[pred+1]  ## pred plus one, beacause classes start from one in dict
    print(sign)
    label.configure(foreground = "#a80758", text = sign)
    
    
    
def classify_button(filepath):
    Clsfy_button =Button(top,text="Classify Image", command = lambda : classify(filepath), borderless=1, padx=15,pady=10)
    Clsfy_button.configure(background="#8c99c2", foreground="#032b6b",font=("montserrat",15))
    #Clsfy_button.place(relx=0.79,rely=0.46)
    
def upload_image():
    try:
        global filepath
        filepath = filedialog.askopenfilename()
        upload = Image.open(filepath)
        upload = upload.resize((300, 300), Image.ANTIALIAS)
        im_up = ImageTk.PhotoImage(upload)
        
        sign_image.configure(image=im_up)
        sign_image.image=im_up
        label.configure(text="",fg="red",font=('montserrat',20, "bold"))
        #classify_button(filepath)
    
    except:
        pass
        

Clsfy_button =Button(top,text="Classify Image", command = lambda : classify(filepath), borderless=1, padx=15,pady=10)
Clsfy_button.configure(background="#8c99c2", foreground="#3b3c40",font=("montserrat",15))
Clsfy_button.pack(side=RIGHT,pady=50)

upload=Button(top,text="Upload Image",command=upload_image, borderless=1, padx=15,pady=10)
upload.configure(background='#8c99c2', foreground='#3b3c40',font=('montserrat',20))
upload.pack(side=BOTTOM,pady=50)

sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="Detect Traffic Sign",pady=20, font=('montserrat',30))
heading.configure(background='#c7dcff',foreground='#3b3c40')
heading.pack()


top.mainloop()

