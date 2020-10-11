import cv2

camera = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("../Resources/haarcascade_frontalface_alt.xml")

name = input("Enter the name of person ")

# Click 20 Pictures and Save them in a Numpy Array (20, 100,100,3) each will contain a face
pics_clicked = 0
cnt = 0
pics = []

while True:
    success, img = camera.read()
    
    if not success:
        continue
        
    #Detect Faces in th eImage
    faces = detector.detectMultiScale(img)
    
    if len(faces)>=1:
        f = faces[0]
        x,y,w,h = f
        green = (0,255,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),green,5)
        #Crop the Face
        crop_face = img[y:y+h, x:x+w]
        cv2.imshow("Crop Face",crop_face)
        
    cv2.imshow("Window Name",img)
    
    if cnt%10==0:
        small_face = cv2.resize(crop_face,(100,100))
        pics.append(small_face)
        pics_clicked += 1
        print("Clicked Picture ",pics_clicked)
        if(pics_clicked==10):
            break
            
    cnt +=1
    
    key = cv2.waitKey(1)
    
    if key==ord('q'):
        print("Stop the Loop")
        break
        
        
import numpy as np
pics = np.array(pics)
print(pics.shape)

np.save(name+".npy",pics)

camera.release()
cv2.destroyAllWindows()


