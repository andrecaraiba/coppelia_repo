import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

import math
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time

'''
readSensorData - It will try to capture the range and angle data from the simulator.
                 The request for the range data is sent in streaming mode to force
                 it to sync with the angle data request which acts as a mutex.

inputs:
    -clientId: simulator client id obtained through a successfull connection with the simulator.
    -range_data_signal_id: string containing the range data signal pipe name.
    -angle_data_signal_id: string containing the angle data signal pipe name.
outputs:
    -returns None if no data is recovered.
    -returns two arrays, one with data range and the other with their angles, if data was 
    retrieved successfully.
'''
DIR = r'./dataset'
cont = 0
img_number = 0
distance_data = []
# Create Dataset
def saveImageData(DIR, filename, img):
    path = os.path.join(DIR, filename)
    cv2.imwrite(path, img)

def saveDistanceData(DIR, filename, distance_data):
    path = os.path.join(DIR, filename)
    with open(path, 'w') as f:
        for item in distance_data:
            f.write("%s\n" % item)



# Rotation image
def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    if rot_point is None:
        rot_point = (width//2, height//2)
    
    rot_mat = cv2.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)


    return cv2.warpAffine(img, rot_mat, dimensions)

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim

if clientID!=-1:
    print ('Connected to remote API server')

    #Get Handle
    returnCode, camHandle = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_oneshot_wait)
    returnCode, blueCuboidHandle = sim.simxGetObjectHandle(clientID, 'Cuboid[1]', sim.simx_opmode_oneshot_wait)
    returnCode, yellowCuboidHandle = sim.simxGetObjectHandle(clientID, 'Cuboid[3]', sim.simx_opmode_oneshot_wait)
    returnCode, redCuboidHandle = sim.simxGetObjectHandle(clientID, 'Cuboid[5]', sim.simx_opmode_oneshot_wait)
    returnCode = 1

    #Get Image
    err, resolution, raw_img = sim.simxGetVisionSensorImage(clientID, camHandle, 0, sim.simx_opmode_streaming)
    #Get Distance
    err, distance_data_blue = sim.simxCheckDistance(clientID, camHandle, blueCuboidHandle, sim.simx_opmode_streaming)
    err, distance_data_yellow = sim.simxCheckDistance(clientID, camHandle, yellowCuboidHandle, sim.simx_opmode_streaming)
    err, distance_data_red = sim.simxCheckDistance(clientID, camHandle, redCuboidHandle, sim.simx_opmode_streaming)
    while(sim.simxGetConnectionId(clientID) != -1):
        #Get Distance
        err, distance_data_blue = sim.simxCheckDistance(clientID, camHandle, blueCuboidHandle, sim.simx_opmode_buffer)
        err, distance_data_yellow = sim.simxCheckDistance(clientID, camHandle, yellowCuboidHandle, sim.simx_opmode_buffer)
        err, distance_data_red = sim.simxCheckDistance(clientID, camHandle, redCuboidHandle, sim.simx_opmode_buffer)
        distance_data = [distance_data_blue, distance_data_yellow, distance_data_red]
        err, resolution, raw_img = sim.simxGetVisionSensorImage(clientID, camHandle, 0, sim.simx_opmode_buffer)
        #distance_data = [3.4, 3.5, 3.8]
        if err == sim.simx_return_ok:
            #print ("image OK!!!")
            img = np.array(raw_img).astype(dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rotated = rotate(rgb, -180)
            flip = cv2.flip(rotated, 1)
            cv2.imshow('image rgb rotated Flip', flip)
            print(distance_data)
            #save image
            if cont % 1000 == 0: 
                saveImageData(DIR, 'image'+str(img_number)+'.png', flip)
                saveDistanceData(DIR, 'image'+str(img_number)+'.txt', distance_data)
                cont = 0
                img_number += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif err == sim.simx_return_novalue_flag:
                print ("no image yet")
                pass
            else:
                #print(err)
                pass
else:
    print ('Failed connecting to remote API server')
     # Parando a simulação     
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)         
        
    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)

print ('Program ended')
cv2.destroyAllWindows()