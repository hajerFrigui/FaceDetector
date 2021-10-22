import logging as log
import datetime as dt
import cv2
#opencv provide the detector model (pretrained data  on bunch of  frontal (no side faces)face images)
#provide "harcascade_frontalface_default" :data files 
 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(trained_face_data)
#classifier : we make a face detector using these information
#classify an object as a face



#import the camera of the pc by value(0) sinon give it the path of the video
#it need waitKey
webcam = cv2.VideoCapture(0)
anterior = 0

while True:
      successful_frame_read, frame = webcam.read()

      grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


      #Detect faces
      face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

      #Draw rectangles
      for (x, y, w, h) in face_coordinates:
          cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 0), 2)

      
      cv2.imshow("Our image",frame)

      #without waitKey it won't display
      key = cv2.waitKey(1)
       
      #stop if Q Key or q key is pressed
      if key==81 or key==113:
           break

webcam.release()     
