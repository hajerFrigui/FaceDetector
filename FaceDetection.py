import cv2
#opencv provide the detector model (pretrained data  on bunch of  frontal (no side faces)face images)
#provide "harcascade_frontalface_default" :data files 
 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(trained_face_data)
#classifier : we make a face detector using these information
#classify an object as a face



#import an image to detect faces in
img = cv2.imread('faces.jpg')


#this classifier take gray scale image 
#convert from color to gray(insterd of 3(red,green,blue) number on take 1 number:gray)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Detect_Faces(coordinates of the rectange surrouded the face)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates) 
#it returns [  [], [], [] , ...    ]: corrdinates of all the faces in the image
#(x, y, w, h) = face_coordinates[0] : coordinate of the first face in the img
#[[248 72 186 186], [], ...] : 248 72 are the top left point  of the rectangle
#186 186 : width and height of the rectangle


#Draw the rectangle
#cv2.rectangle(img, (248, 72), (248+186, 72+186), (0, 255, 0), 2)
#(0,255,0): color of the rectangle 
#2: thikness of the rectangle

#looping all the  faces and draw a rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+h, y+h), (0, 255, 0), 2)


#show the image and close it immadiatly and go to the "print("hello")""
cv2.imshow("Our image",img)
#so we add this line to keep it open and wait until you press any key 
cv2.waitKey()

print("hellooo")