from PIL import Image
import face_recognition
import datetime
import cv2
import imutils
import os
import os.path
import numpy as np
import multiprocessing as mp
from time import sleep
from face_recognition.face_recognition_cli import image_files_in_folder
video_capture = cv2.VideoCapture(0)
# Load the jpg file into a numpy array
train_dir = "scs"
process_this_frame = False
# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_encodings = []
known_face_encodings = []
known_face_names = []
face_names = []
firstFrame = None
min_area = 1500
folder_list = []
NUM_WORKERS = mp.cpu_count()
# Get directory folders and split for workers
for class_dir in os.listdir(train_dir): 
    if not os.path.isdir(os.path.join(train_dir, class_dir)):
       continue
    folder_list.append(class_dir)
person_list = np.array_split(folder_list, NUM_WORKERS)
 
# Loop through each person in the training set  
def mp_process(person_list): 
	for class_dir in person_list:
	    # Loop through each training image for the current person
	    for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
	        
	        image = face_recognition.load_image_file(img_path)
	        face_bounding_boxes = face_recognition.face_locations(image, model='hog')
	
	        if len(face_bounding_boxes) != 1:
	            # If there are no people (or too many people) in a training image, skip the image.
	            print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
	                face_bounding_boxes) < 1 else "Found more than one face"))
	        else:
	            # Add face encoding for current image to the training set
	            known_face_encodings.append(face_recognition.face_encodings(
	                image, known_face_locations=face_bounding_boxes, num_jitters=2)[0])
	            known_face_names.append(class_dir)
	#print(known_face_names)
	return known_face_encodings,known_face_names

pool = mp.Pool(NUM_WORKERS)
s = pool.map(mp_process, person_list)
print('Active children count: %d ' %len(mp.active_children()))
pool.close()
pool.join()
known_face_encodings = s[0][0]
known_face_names = s[0][1]
known_face_encodings.extend(s[1][0])
known_face_names.extend(s[1][1])
print("Training done")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
#    cv2.rectangle(frame, (323,205), (888, 720), (0, 255, 0), 2)
#    small_frame = frame[205:720, 323:888]
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    #rgb_small_frame = small_frame[:, :, ::-1]
    # convert to grayscale, and blur it
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    	# if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray_frame       
        firstFrame_time = datetime.datetime.now()
        continue
    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray_frame)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
           continue    
        process_this_frame = True
        #ret, frame = video_capture.read()
        # compute the bounding box for the contour, crop frame and to it size
        (x, y, w, h) = cv2.boundingRect(c)
 #       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        #print("Motion detected")
    #Counter for first frame null'ing
    time_delta = datetime.datetime.now() - firstFrame_time
    if time_delta.seconds > 30:
        firstFrame = None
        
    
    if process_this_frame:
        sleep(0.5)
        
        ret, frame = video_capture.read()
        small_frame = frame[y:y + h, x:x + w]
        #Applying Histogram Equalization B&W
        #small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY )
        #small_frame = cv2.equalizeHist(small_frame)
        #small_frame = cv2.fastNlMeansDenoising(small_frame,None,10,7,21)
        #Applying Erosion
        #kernel = np.ones((5,5),np.uint8)
        #frame = cv2.erode(frame, kernel, iterations=1)  
        #frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB )
        
        # Denoising with color --WORSE--
        #frame = cv2.fastNlMeansDenoisingColored(small_frame,None,10,10,7,21)
        
 		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)     
        rgb_frame = small_frame[:, :, ::-1]
       
		# Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(
            rgb_frame, number_of_times_to_upsample=1, model='hog')
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=2)
            #	        print("I found {} face(s) in this photograph at {}.".format(len(face_locations),datetime.datetime.now()))
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
                print("I found {} face at {}.".format(name, datetime.datetime.now()))

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(small_frame, (left, top),(right, bottom), (0, 0, 255), 2)
                cv2.rectangle(small_frame, (left, bottom),(right, bottom + 35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(small_frame, name, (left , bottom + 33),font, 1.0, (255, 255, 255), 1)
            out_frame = small_frame[:, :, ::-1]
            pil_image = Image.fromarray(out_frame)
            file = datetime.datetime.now()
            ext = ".jpg"
            pil_image.save(str(file)+str(ext),quality=100)
        process_this_frame = False
#    sleep(0.4)


# Release handle to the webcam
video_capture.release()

