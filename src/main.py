import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

constant_value = 0.5

skip_multiple_faces = True

def cropface(image):
    img = cv2.imread(image)
    height, width, channels = img.shape 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if(len(faces) != 1 and skip_multiple_faces):
        print(f"More than one face found, skipping... {image}")
        return
    
    print(f"Found {len(faces)} face(s) in {image}...")
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
        output = f"../output/{image.split('/')[-1].split('.')[0]}_{idex}.jpg"
        cv2.imwrite(output, faces)

if __name__ == "__main__":
    dir_path = "../input"
    res = os.listdir(dir_path)
    for file in res:
        cropface(f"{dir_path}/{file}")
