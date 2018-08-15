
# coding: utf-8

# In[1]:


#importing all required libraries 
import cv2
import os
import numpy as np


# In[2]:


#I will be training my model to recognize 3 persons(ME and my 2 friends)
#List persons will store the name of person at index 1,2 and 3.I have created folders as 1,2,3 in training dataset for respective faces
persons=['None','Sarthak','Akash','Ajay']
#training data address
train_data_path=r'D:\training-data'


# In[3]:


#detect_face function will detect the face using opencv
def detect_face(img):
    #Converting image to grayscale for opencv 
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #I will use LBP cascade as it is fast.There are other more accurate but slow.
    #This is located in opencv\data
    f_cascade = cv2.CascadeClassifier(r'C:\Program Files (x86)\opencv\sources\data\lbpcascades\lbpcascade_frontalface_improved.xml')
    #Detecting face using MultiScale Funct.Result is a list of faces
    faces = f_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5);
    #If no faces in the image.
    if (len(faces) == 0):
        return None,None
    #Assuming only potrait photos
    (x, y, w, h) = faces[0]
    #returning the face part and coordinates 
    return gray_img[y:y+w, x:x+h],faces[0]
    


# In[4]:


#loading training data
def training_data(train_data_path):
    #faces and labels will have face details and corresponding labels respectively.
    faces=[]
    labels=[]
    #using os function to load folder names
    folders= os.listdir(train_data_path)
    for folder_name in folders:
        label=int(folder_name)
        person_path=train_data_path + "/" + folder_name
    
        person_images_names = os.listdir(person_path)
        for image_name in person_images_names:
            #ignore system files with .
            if image_name.startswith("."):
                continue;

            image_path = person_path + "/" + image_name
            #read image using imread function of opencv
            image = cv2.imread(image_path)
            face,rectangle_coordinates=detect_face(image)
            if face is not None:
                #adding faces and corresponding labels
                faces.append(face)
                labels.append(label)
        
    return faces,labels
                
                


# In[5]:


#creating LBPH face recognizer 
face_recognizer =cv2.face.LBPHFaceRecognizer_create()


# In[6]:


#train our face recognizer of our training faces
faces, labels =training_data(train_data_path)
face_recognizer.train(faces, np.array(labels))


# In[7]:


#image recognition using predict funct
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detecting face from the image
    #Converting image to grayscale for opencv 
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #I will use LBP cascade as it is fast.There are other more accurate but slow.
    #This is located in opencv\data
    f_cascade = cv2.CascadeClassifier(r'C:\Program Files (x86)\opencv\sources\data\lbpcascades\lbpcascade_frontalface_improved.xml')
    #Detecting face using MultiScale Funct.Result is a list of faces
    test_faces = f_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5);
    #If no faces in the image.
    if (len(test_faces) == 0):
        return None
    
    for (x, y, w, h) in test_faces:
        test_face=(gray_img[y:y+w, x:x+h])
        #predict the image using our face recognizer 
        label= face_recognizer.predict(test_face)
        #get name of respective label returned by face recognizer
        label_name = persons[label[0]]
        #function to draw rectangle on the image and write label to it
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img,label_name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX ,3, (0, 255, 0),5)
        
    return img


# In[16]:


print("Predicting images...")
#load test image
test_img1 = cv2.imread(r'D:\test-data\MIX\3.jpg')
#perform a prediction
predicted_img1 = predict(test_img1)
#Resizing window
predicted_img1= cv2.resize(predicted_img1, (700,700))
print("Prediction complete")
#display image using opencv imshow funct
cv2.imshow('Test Imag',predicted_img1) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

