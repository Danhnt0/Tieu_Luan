
import cv2
import numpy as np
import tensorflow as tf

image = cv2.imread("Image/test2.jpg")
#image = cv2.resize(image,(192,79))
copy = image.copy()

im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]

#load model
adv_model = tf.keras.models.load_model("Model/advs_model.h5")
base_model = tf.keras.models.load_model("Model/base_model.h5")

adv = []
base = []

for i in contours:
    #if cv2.contourArea(i) > 400:
        (x,y,w,h) = cv2.boundingRect(i)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        subImage = thre[y:y+h,x:x+w]
        subImage = np.pad(subImage,(20,20),'constant',constant_values=(0,0))
        subImage = cv2.resize(subImage, (28, 28), interpolation=cv2.INTER_AREA)
        subImage = cv2.dilate(subImage, (3, 3))

        img = subImage.reshape(1,28,28,1)
        img = img/255.0
        img = img.astype(np.float32)
        
        adv_pred = adv_model.predict(img)
        adv_pred = np.argmax(adv_pred)
        adv.append(int(adv_pred))
        cv2.putText(copy,str(int(adv_pred)),(x,y+30),0,1,(0,0,255),2)

        base_pred = base_model.predict(img)
        base_pred = np.argmax(base_pred)
        base.append(base_pred)
        cv2.putText(copy,str(int(base_pred)),(x,y+60),0,1,(0,255,0),2)

#reverse the list
#adv = adv[::-1]
#base = base[::-1]

print("adv_pred",*adv)
print("base_pred: ",*base)

cv2.imshow("image",copy)
cv2.waitKey(0)


   
