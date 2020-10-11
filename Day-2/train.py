import cv2

camera = cv2.VideoCapture(0)


name = input("Enter the name of person ")


while True:
    success, img = camera.read()
    
    if not success:
        continue
    
    cv2.imshow("Window Name",img)
    key = cv2.waitKey(1)
    if key==ord('q'):
        print("Stop the Loop")
        break
        
        
camera.release()
cv2.destroyAllWindows()