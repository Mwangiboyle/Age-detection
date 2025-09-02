# %%
import cv2

faceProto='opencv_face_detector.pbtxt' #loadign the face detector
faceModel='opencv_face_detector_uint8.pb' #loading the facedetector unit
ageProto='age_deploy.prototxt' #loading thr prototxt file
ageModel='age_net.caffemodel' #loading the caffemodel
genderProto='gender_deploy.prototxt' #loading the gender deploy library
genderModel='gender_net.caffemodel' #loading the gender net cafee model


def detectFace(net,frame,confidence_threshold=0.7):
    """A function that will detect the face in the video camera.
It will be using the DNN and OpenCV for the camera.
Then it will create a boundary around the face and will detect the face.
Then it will use the Caffe model files to detect the gender and the
age of the person."""
    frameOpencvDNN=frame.copy()
    frameHeight=frameOpencvDNN.shape[0]
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
    return frameOpencvDNN,faceBoxes

gender_List=['Male','Female'] #making the list of gender
age_List=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)'] #making the list of range of ages
face_model=cv2.dnn.readNet(faceModel,faceProto) #loading the faceNet model
age_model=cv2.dnn.readNet(ageModel,ageProto) #loading the agenet model
gender_model=cv2.dnn.readNet(genderModel,genderProto)  #loading the gender model

video=cv2.VideoCapture(0) #starting the video camera so that we can detect a face
padding=20 #we are keeping a padding of 20 because that would look optimal

while cv2.waitKey(1)<0: #starting while loop to run the video camera and trying to detect a face.
    hasFrame,frame=video.read() #read function for reading video
    if not hasFrame:     #checking the video  has a frame in it or not
        cv2.waitKey() #wait key of OpenCV function, we are using a waitKey function of OpenCV
        break #breaking the loop when no frame is found

    resultImg,faceBoxes=detectFace(face_model,frame) #calling the detectface function and passing the faceNet and frame object into the function

    for faceBox in faceBoxes: #for multiple faces in the video we are using a for loop
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        gender_model.setInput(blob)  #passing the blob into setInput
        genderPreds=gender_model.forward() #calling the forward fucntion
        gender=gender_List[genderPreds[0].argmax()] #loading the gender list

        age_model.setInput(blob) #passing the blob into setInputs
        agePreds=age_model.forward() #calling the forward function
        age=age_List[agePreds[0].argmax()] #loading the age list
        cv2.putText(resultImg,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA) #marking the gender and age
        cv2.imshow("Gender and Age",resultImg) #showing the image we are printing out the result here of the gender and the age.


        if cv2.waitKey(33) & 0xFF == ord('q'): #break loop when closing or there is no frame left or we press the q button
            break

cv2.destroyAllWindows() #close the running window of OpenCV
