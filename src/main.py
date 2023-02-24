import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

constant_value = 0.5

resize_image = False

resize_dimension = 512

def cropface(image):
    img = cv2.imread(image)
    height, width, channels = img.shape 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for idex, (x, y, w, h) in enumerate(faces):
        hp_1 = x
        hp_2 = (width - (x+w))
        vp_1 = y
        vp_2 = (height - (y+h))

        padding_x = hp_1
        padding_y = vp_1
        padding = 100
        if(hp_1 > hp_2):
            padding_x = hp_2
        if(vp_1 > vp_2):
            padding_y = vp_2
        
        if (padding_x > padding_y):
            padding = padding_y
        else:
            padding = padding_x

        if(padding >= w * constant_value):
            padding = int(w * constant_value)

        if(padding %2 != 0):
            padding = padding - 1

        

        faces = img[(y - padding):(y + h + padding), (x - padding):(x + w + padding)]
        gray = gray[(y - padding):(y + h + padding), (x - padding):(x + w + padding)]
        gray_faces = eye_cascade.detectMultiScale(gray)
        if(not len(gray_faces)):
            print(f"There is no eye on this face OMG, skipping {image}")
            return
        output = f"../output/{idex}_{image.split('/')[-1]}"
        if(resize_image):
            cv2.imwrite(output, cv2.resize(faces, (resize_dimension, resize_dimension)))
        else:
            cv2.imwrite(output, faces)

if __name__ == "__main__":
    dir_path = "../input"
    res = os.listdir(dir_path)
    for file in res:
        cropface(f"{dir_path}/{file}")
