# Face Detection Lock 

import cv2 
import numpy as np

face_classfier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_classfier.detectMultiScale(gray , 1.3 , 5)
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h , x:x+w]
        
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret , frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame) , (200 , 200))
        face = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)
        
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path , face)
        
        cv2.putText(face , str(count) , (50,50) , cv2.FONT_HERSHEY_COMPLEX , 1  , (0,255,0) , 2)
        cv2.imshow('face detector' , face)
        
    else:
        print("Face not found")
        pass
    
    if cv2.waitKey(1) == 13 or count == 100:
        break
        
cap.release()
cv2.destroyAllWindows()
print("collecting samples complete")

import cv2
import numpy as np
from os import listdir
from os.path import isfile , join

data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path , f))]

training_Data , Labels = [] , []


for i , files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)
    training_Data.append(np.asarray(images , dtype=np.uint8))
    Labels.append(i)
    
Labels = np.asarray(Labels , dtype=np.int32)

model = cv2.face_LBPHFaceRecognizer.create()

model.train(np.asarray(training_Data) , np.asarray(Labels))
print("model trained successfully")

def face_detector(img , size=0.5):
    
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_classfier.detectMultiScale(gray , 1.3 , 5)
    if faces is ():
        return img , []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,255,255) , 2)
        roi = img[y:y+h , x:x+w]
        roi = cv2.resize(roi , (200 , 200))
    
    return img , roi 

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    image , face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1]/400)) )
            display_string = str(confidence) + '%confident it is user'
            
        cv2.putText(image , display_string , (100,120) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,120,150 , 2))
        
        if confidence > 75:
            cv2.putText(image , "unlocked" , (250,450) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) ,2)
            cv2.imshow('face recognition' , image)
            
        else:
            cv2.putText(image , "locked" , (250,450) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) ,2)
            cv2.imshow('face recognition' , image)
            
    except:
            cv2.putText(image , "no face found" , (250,450) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) ,2)
            cv2.putText(image , "locked" , (250,450) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) ,2)
            cv2.imshow('face recognition' , image)
            pass
    
    if cv2.waitKey(1) == 13:
        break
        

cap.release()
cv2.destroyAllWindows()

#Face detection mail
# task 6.1



# importing all the useful libraries for the task 6
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import getpass
import subprocess

# sending the email to the person whose face is detected 
def sendMail(receiver_address , name):
    #The mail addresses and passwordharsh
    sender_address = 'harshadamohite1762000@gmail.com'
    sender_pass = getpass.getpass(prompt="Enter your password")
    
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] =sender_address 
    message['To'] = receiver_address
    message['Subject'] = 'We have detected you and your name is:' + name
    
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print("mail sent to the address")

sendMail("harshamohite179@gmail.com","harshada")

# Task 6.2 



def aws_ec2_ebs():
    # AWS cli command for creating aws ec2 instance

    # first - creating the key-pair for the aws 
    subprocess.getoutput("aws ec2 create-key-pair --key-name aws_key")

    # second  - creating security group for our ec2 instance 
    subprocess.getoutput("aws ec2 create-security-group --group-name MyAWSSecurityGroup --description "My AWS security group"")

    # third  - creating/ launching an ec2 instance
    subprocess.getoutput("aws ec2 run-instances --image-id ami-0e306788ff2473ccb --instance-type t2.micro --key-name aws_key --security-group-ids sg-0d044f5752b4e9322")

    # fourth - creating an ebs volume in aws 
    subprocess.getoutput("aws ec2 create-volume --volume-type gp2  --size 1  --availability-zone  ap-south-1a")

    # fifth - Attaching the above created EBS volume to the instance
    subprocess.getoutput("aws ec2 attach-volume --volume-id vol-07cde9d02dea697d3 --instance-id i-0ef7886cf2396a580  --device /dev/sdf")



