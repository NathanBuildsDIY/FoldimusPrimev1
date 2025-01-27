#control for laundry folding proof of concept

# import the necessary packages
import math
import numpy as np
import os
from glob import glob
import time
from datetime import datetime
from gpiozero import Device, PhaseEnableMotor, Robot, PhaseEnableRobot, LED, Servo, AngularServo, Button, DistanceSensor
import tflite_runtime.interpreter as tfi
import cv2
from picamera2 import Picamera2, Preview

#Adjust operations with these parms:
classnames = ["dad","daughter"]

#define the servos
#factory = PiGPIOFactory() #not available on pi 5 :(
clothespin1 = AngularServo(20, initial_angle=120, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 
clothespin2 = AngularServo(21, initial_angle=120, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 

foldHorizontal1 = AngularServo(18, initial_angle=0, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 
foldHorizontal2 = AngularServo(17, initial_angle=180, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 
foldHorizontal3 = AngularServo(15, initial_angle=0, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 
foldHorizontal4 = AngularServo(14, initial_angle=180, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 
foldVertical = AngularServo(27, initial_angle=30, min_angle=0, max_angle=180, min_pulse_width=5/10000, max_pulse_width=28/10000) 

#define the linear actuators
motorDir=PhaseEnableMotor(9,10) #9=dir for all motors
foldArm=PhaseEnableMotor(11,1) #dir,pwm
armInOut=PhaseEnableMotor(0,7) #forward = out, backward = in
armUpDown=PhaseEnableMotor(13,8) #forward = up, backward = down
lineMove=PhaseEnableMotor(6,26) #backward = move clothes into place. forward fails for some reason
sortingRamp=PhaseEnableMotor(5,19) #backward = retract, forward = push sorting ramp out

#create the image classifier
interpreter = tfi.Interpreter(model_path='/home/gibby/laundry/model_reg_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#create run directory to save photos and go there
laundrydir = os.path.expanduser("~/laundry/log/")
date = datetime.now().strftime("%m_%d_%I_%M_%S_%p_%f")
rundir = laundrydir+date
os.makedirs(rundir)
os.chdir(rundir)

#define the camera
camwidth = 2464 
camheight = 3280 
Xtarget = camwidth*.5 #where we want the center of bright targeting array to end up
Ytarget = camheight*.5
CloseEnough = 100 #number of pixels we'll allow to be "off" between true center and center of target array
#create camera 
picam = Picamera2()
picam.set_controls({"ExposureTime": 1000, "AnalogueGain": 8.0})
config = picam.create_still_configuration()
picam.configure(config)
picam.start()

# Visualization parameters for text on classification images
_ROW_SIZE = 10  # pixels
_LEFT_MARGIN = 10  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 0.5
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10

#define the sensors - none for now

#useful functions
def takePhoto(prepend):
  time.sleep(.1) #let things settle before we snap a photo. hopefully we get clearer images this way
  #snapshot from camera and save. assumes we're already in correct directory
  date = datetime.now().strftime("%m_%d_%I_%M_%S_%p_") #note: keep it short and don't allow : in filename - error
  ident = datetime.now().strftime("%f")
  date=date+ident
  photo_name = prepend + "_" + date + ".jpg"
  picam.capture_file(photo_name)
  image = cv2.imread(photo_name)
  # rotate ccw - that makes it right side up. Camera is 90 position so that larger pixels (3280) capture vertical.
  out=cv2.transpose(image)
  out=cv2.flip(out,flipCode=0)
  cv2.imwrite(photo_name,out)
  return photo_name
def move1Motor(motor,currentAngle,targetAngle):
  #move servos gradually to avoid jerking the robot to pieces
  minorAdjust=5
  if targetAngle-currentAngle > 0:
    minorAdjust=minorAdjust
  else:
    minorAdjust=-minorAdjust
  while abs(targetAngle-currentAngle)>0 and 0<=currentAngle+minorAdjust<=180:
    currentAngle=currentAngle+minorAdjust
    motor.angle=currentAngle
    time.sleep(0.05)
  return currentAngle
def move2Motors(motor1,currentAngle1,targetAngle1,motor2,currentAngle2,targetAngle2):
  #move servos gradually to avoid jerking the robot to pieces
  minorAdjust=5
  minorAdjust1=0
  if targetAngle1-currentAngle1 > 0:
    minorAdjust1=minorAdjust
  else:
    minorAdjust1=-minorAdjust
  minorAdjust2=0
  if targetAngle2-currentAngle2 > 0:
    minorAdjust2=minorAdjust
  else:
    minorAdjust2=-minorAdjust
  while (abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180) or (abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180):
    if abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180:
      currentAngle1=currentAngle1+minorAdjust1
      motor1.angle=currentAngle1
    if abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180:
      currentAngle2=currentAngle2+minorAdjust2
      motor2.angle=currentAngle2
    time.sleep(0.02)
  return currentAngle1, currentAngle2
def move4Motors(motor1,currentAngle1,targetAngle1,motor2,currentAngle2,targetAngle2,motor3,currentAngle3,targetAngle3,motor4,currentAngle4,targetAngle4):
  #move servos gradually to avoid jerking the robot to pieces
  minorAdjust=5
  minorAdjust1=0
  if targetAngle1-currentAngle1 > 0:
    minorAdjust1=minorAdjust
  else:
    minorAdjust1=-minorAdjust
  minorAdjust2=0
  if targetAngle2-currentAngle2 > 0:
    minorAdjust2=minorAdjust
  else:
    minorAdjust2=-minorAdjust
  minorAdjust3=0
  if targetAngle3-currentAngle3 > 0:
    minorAdjust3=minorAdjust
  else:
    minorAdjust3=-minorAdjust
  minorAdjust4=0
  if targetAngle4-currentAngle4 > 0:
    minorAdjust4=minorAdjust
  else:
    minorAdjust4=-minorAdjust
  while (abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180) or (abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180) or (abs(targetAngle3-currentAngle3)>0 and 0<=currentAngle3+minorAdjust3<=180) or (abs(targetAngle4-currentAngle4)>0 and 0<=currentAngle4+minorAdjust4<=180):
    if abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180:
      currentAngle1=currentAngle1+minorAdjust1
      motor1.angle=currentAngle1
    if abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180:
      currentAngle2=currentAngle2+minorAdjust2
      motor2.angle=currentAngle2
    if abs(targetAngle3-currentAngle3)>0 and 0<=currentAngle3+minorAdjust3<=180:
      currentAngle3=currentAngle3+minorAdjust3
      motor3.angle=currentAngle3
    if abs(targetAngle4-currentAngle4)>0 and 0<=currentAngle4+minorAdjust4<=180:
      currentAngle4=currentAngle4+minorAdjust4
      motor4.angle=currentAngle4
    time.sleep(0.02)
    #print(currentAngle1," ",currentAngle2," ",currentAngle3," ",currentAngle4)
  return currentAngle1, currentAngle2, currentAngle3, currentAngle4
def move5Motors(motor1,currentAngle1,targetAngle1,motor2,currentAngle2,targetAngle2,motor3,currentAngle3,targetAngle3,motor4,currentAngle4,targetAngle4,motor5,currentAngle5,targetAngle5):
  #move servos gradually to avoid jerking the robot to pieces
  minorAdjust=5
  minorAdjust1=0
  if targetAngle1-currentAngle1 > 0:
    minorAdjust1=minorAdjust
  else:
    minorAdjust1=-minorAdjust
  minorAdjust2=0
  if targetAngle2-currentAngle2 > 0:
    minorAdjust2=minorAdjust
  else:
    minorAdjust2=-minorAdjust
  minorAdjust3=0
  if targetAngle3-currentAngle3 > 0:
    minorAdjust3=minorAdjust
  else:
    minorAdjust3=-minorAdjust
  minorAdjust4=0
  if targetAngle4-currentAngle4 > 0:
    minorAdjust4=minorAdjust
  else:
    minorAdjust4=-minorAdjust
  minorAdjust5=0
  if targetAngle5-currentAngle5 > 0:
    minorAdjust5=minorAdjust
  else:
    minorAdjust5=-minorAdjust
  while (abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180) or (abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180) or (abs(targetAngle3-currentAngle3)>0 and 0<=currentAngle3+minorAdjust3<=180) or (abs(targetAngle4-currentAngle4)>0 and 0<=currentAngle4+minorAdjust4<=180) or (abs(targetAngle5-currentAngle5)>0 and 0<=currentAngle5+minorAdjust5<=180):
    if abs(targetAngle1-currentAngle1)>0 and 0<=currentAngle1+minorAdjust1<=180:
      currentAngle1=currentAngle1+minorAdjust1
      motor1.angle=currentAngle1
    if abs(targetAngle2-currentAngle2)>0 and 0<=currentAngle2+minorAdjust2<=180:
      currentAngle2=currentAngle2+minorAdjust2
      motor2.angle=currentAngle2
    if abs(targetAngle3-currentAngle3)>0 and 0<=currentAngle3+minorAdjust3<=180:
      currentAngle3=currentAngle3+minorAdjust3
      motor3.angle=currentAngle3
    if abs(targetAngle4-currentAngle4)>0 and 0<=currentAngle4+minorAdjust4<=180:
      currentAngle4=currentAngle4+minorAdjust4
      motor4.angle=currentAngle4
    if abs(targetAngle5-currentAngle5)>0 and 0<=currentAngle5+minorAdjust5<=180:
      currentAngle5=currentAngle5+minorAdjust5
      motor5.angle=currentAngle5
    time.sleep(0.02)
    #print(currentAngle1," ",currentAngle2," ",currentAngle3," ",currentAngle4)
  return currentAngle1, currentAngle2, currentAngle3, currentAngle4, currentAngle5
def findImageDifferences(before,after):
  image1 = cv2.imread(before)
  image2 = cv2.imread(after)
  # Ensure both images have the same dimensions
  if image1.shape != image2.shape:
    raise ValueError("The images must have the same dimensions.")
  # Compute the absolute difference between the two images
  difference = cv2.absdiff(image1, image2)
  # Threshold the difference to identify significant changes
  _, thresholded_diff = cv2.threshold(difference, 70, 255, cv2.THRESH_BINARY)
  #find average x,y position of non-white pixels - first turn any colored pixels black. then average black pixels
  # Convert the image to grayscale (optional, depends on the image)
  gray_image = cv2.cvtColor(thresholded_diff, cv2.COLOR_BGR2GRAY)
  # Create a mask where non-black pixels are 1 and black pixels are 0
  # A pixel is considered non-black if its intensity is greater than 0
  non_black_pixels = gray_image > 0
  # Get the coordinates (x, y) of non-black pixels
  y_indices, x_indices = np.where(non_black_pixels)
  # Calculate the average x and y coordinates
  average_x = np.mean(x_indices)
  average_y = np.mean(y_indices)
  # Print the average (x, y) position
  print(f"Average X: {average_x}, Average Y: {average_y}")
  # Draw a circle at the average position on the original image
  cv2.circle(thresholded_diff, (int(average_x), int(average_y)), 15, (0, 255, 0), -1)
  cv2.putText(thresholded_diff, str(int(average_x))+" "+str(int(average_y)), (int(average_x)-20, int(average_y)-20),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 4)
  cv2.imwrite("thresh_"+after, thresholded_diff)
  return(int(average_x),int(average_y))
def moveESCMotor(direction,seconds,ESCmotor,speed):
  if direction == "backward": motorDir.backward(1); 
  if direction == "forward": motorDir.forward(1)
  ESCmotor.forward(speed); time.sleep(seconds); ESCmotor.stop()
  time.sleep(0.1)
def preprocess_image_uint8_resize_norm(image_path, input_shape):
  # Function to preprocess image (resize, normalize to [-128, 127])
  # Load the image
  image = cv2.imread(image_path)
  # Resize the image to match the input size of the model
  image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
  image_normalized = image_resized
  # Convert the image to floating point format and normalize to [0, 255]
  image_normalized = image_resized.astype(np.float32)
  # Quantize the image: Map [0, 255] -> [-128, 127] for int8 quantization
  image_normalized -= 128
  image_normalized = np.clip(image_normalized, -128, 127)  # Clip to avoid overflow
  image_normalized = image_normalized.astype(np.uint8)
  # Add a batch dimension (as models expect [batch_size, height, width, channels])
  image_batch = np.expand_dims(image_normalized, axis=0)
  return image_batch
def classifyImage(image_path):
  # Preprocess the image
  input_shape = input_details[0]['shape'] #input details defined with model at start of program
  preprocessed_image_uint8 = preprocess_image_uint8_resize_norm(image_path, input_shape)
  # Set the input tensor
  interpreter.set_tensor(input_details[0]['index'], preprocessed_image_uint8)
  # Run inference
  interpreter.invoke()
  # Get the output tensor
  output_data = interpreter.get_tensor(output_details[0]['index'])
  # Output is a vector of probabilities (for classification tasks). int8 model predictions adds up to 256.
  predictions = output_data[0]
  # Get the index of the highest probability
  predicted_class = np.argmax(predictions)
  #name the class to make it pretty
  classname = classnames[predicted_class]
  # Print the predicted class index and probability
  percentage = int(100*predictions[predicted_class]/256)
  printstr = "Predicted class index: "+classname+" Probability: "+str(percentage)+"%"
  image = cv2.imread(image_path)
  image_path = "classified_"+image_path
  cv2.putText(image, printstr, (20,100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 4)
  print("saving image:",image_path,"with the string:",printstr)
  cv2.imwrite(image_path,image)
  return predicted_class
def testAllMotors():
  #Check Motor Functions
  #3 axis test & sorting ramp test
  motorDir.forward(1); #set dir
  foldArm.forward(1); time.sleep(1); foldArm.stop();
  armInOut.forward(1); time.sleep(1); armInOut.stop();
  armUpDown.forward(1); time.sleep(1); armUpDown.stop();
  sortingRamp.forward(1); time.sleep(1); sortingRamp.stop(); 
  motorDir.backward(1); 
  foldArm.backward(1); time.sleep(1.5); foldArm.stop();
  armInOut.backward(1); time.sleep(1.5); armInOut.stop();
  armUpDown.backward(1); time.sleep(1.5); armUpDown.stop();
  sortingRamp.backward(1); time.sleep(1.5); sortingRamp.stop(); 
  time.sleep(1)
  #line test
  motorDir.backward(1); 
  lineMove.backward(0.1); time.sleep(0.2); lineMove.stop()
  time.sleep(1)
  #clothespintest
  move2Motors(clothespin1,120,140,clothespin2,110,140)
  time.sleep(1)
  move2Motors(clothespin1,140,120,clothespin2,140,110)
  time.sleep(1)
  #folding arm test
  move4Motors(foldHorizontal1,0,180,foldHorizontal2,180,0,foldHorizontal3,0,180,foldHorizontal4,180,0)
  move1Motor(foldVertical,30,180)
  time.sleep(1)
  move5Motors(foldHorizontal1,180,160,foldHorizontal2,0,20,foldHorizontal3,180,160,foldHorizontal4,0,20,foldVertical,180,160)
  time.sleep(0.2) #time for clothes to drop
  move1Motor(foldVertical,160,30)
  move4Motors(foldHorizontal1,160,0,foldHorizontal2,20,180,foldHorizontal3,160,0,foldHorizontal4,20,180)
  time.sleep(1)
  print("finished initial motor tests")

#normal operation loop
initialImage=takePhoto("initial")
#initial move of laundry line
input("Initial photo taken. Press enter when you're ready to move line and take photos to center it")
moveESCMotor("backward",5,lineMove,0.1)
#loop to center the clothing - pay attention to x and try to center in image
currentX = 0
currentY = 0
currentImage=""
while currentX < Xtarget-CloseEnough-100:
  moveESCMotor("backward",1,lineMove,0.1)
  currentImage=takePhoto("current")
  currentX,currentY = findImageDifferences(initialImage,currentImage)
print("garment is centered on X axis, now classify")
predicted_class=classifyImage(currentImage)
#predicted_class=1 #for testing daughter class before model ready
print("predicted class:", predicted_class)

print("push out arm in prep to hang on it")
moveESCMotor("forward",60,foldArm,1) #push out fold arm fully

#move arm up to 60% midpoint. It's already 12 seconds from bottom (that's height of wooden arm). 
#1800 pixels is "neutral" start pt. 50 seconds up is 800 pixels. Travel is 1000 pixels
moveY = 2000-currentY #pixels from "neutral" position
moveDir = "forward"
if moveY < 0: moveDir = "backward"
moveUpDownSeconds = int(moveY* 60/1000)
if moveUpDownSeconds > 55: moveUpDownSeconds = 55 #don't go past here, end of ESC
if moveUpDownSeconds < -12: moveUpDownSeconds = -12 #can't go past the lowest point
if moveUpDownSeconds < 0: moveUpDownSeconds = -moveUpDownSeconds #we already handled direction above, need positive seconds
print("moving up/down to center on fold line:",moveUpDownSeconds," moveDir: ",moveDir)
moveESCMotor(moveDir,moveUpDownSeconds+1,armUpDown,1)

print("press enter when ready to move in arm assembly")
moveESCMotor("backward",39,armInOut,1)

input("wait to start clothespins")
#drop clothing with clothespins
move2Motors(clothespin1,110,140,clothespin2,110,170)
time.sleep(3)
move2Motors(clothespin1,110,120,clothespin2,170,110)
time.sleep(1)

print("Garment is folded over arm and ready to press up to folding arm")
#Move assembly 7 seconds out so we don't catch metal arm on wooden one
moveESCMotor("forward",15,armInOut,1)
#reverse direction of up/down before moving it. Same time as above
if moveDir == "forward": moveDir = "backward"; 
elif moveDir == "backward": moveDir = "forward"; 
moveESCMotor(moveDir,moveUpDownSeconds,armUpDown,1)

print("check arm is down far enough - lined up with wooden folding arm before pulling in")
#move assembly in 7 seconds to meet metal/wooden
moveESCMotor("backward",15,armInOut,1)

input("check lined up, ready for pinchers to close and fold")
move4Motors(foldHorizontal1,0,180,foldHorizontal2,180,0,foldHorizontal3,0,180,foldHorizontal4,180,0)
time.sleep(0.5)
moveESCMotor("backward",62,foldArm,1)

print("check ready to fold up clothes and release")
move1Motor(foldVertical,30,180)
time.sleep(1)

print("3rd fold - move assembly down 13 sec")
moveESCMotor("backward",13,armUpDown,1)
print("3rd fold - move assembly in 7 sec")
moveESCMotor("backward",7,armInOut,1)
print("3rd fold - move arm out 30 sec")
moveESCMotor("forward",35,foldArm,1)
moveOutSeconds = 39-moveUpDownSeconds/3 #the further it moved up, the smaller the garment. 
#we need all of it - just make it 39+7.
moveOutSeconds = 39
print("3rd fold - move assembly out",str(moveOutSeconds),"plus 7")
moveESCMotor("forward",moveOutSeconds+7,armInOut,1)


input("enter to drop the clothes")
move5Motors(foldHorizontal1,180,0,foldHorizontal2,0,180,foldHorizontal3,180,0,foldHorizontal4,0,180,foldVertical,180,140)
time.sleep(1.5)
move5Motors(foldHorizontal1,180,0,foldHorizontal2,0,180,foldHorizontal3,180,0,foldHorizontal4,0,180,foldVertical,180,140)
time.sleep(4.5)
move1Motor(foldVertical,140,30)

print("3rd fold - move assembly up 13 seconds")
moveESCMotor("forward",13,armUpDown,1)

print("3rd fold - move assembly in:",str(moveOutSeconds))
moveESCMotor("backward",moveOutSeconds,armInOut,1)

input("check lined up, ready for pinchers to close and fold, then move arm out 35 seconds")
move4Motors(foldHorizontal1,0,180,foldHorizontal2,180,0,foldHorizontal3,0,180,foldHorizontal4,180,0)
time.sleep(0.5)
moveESCMotor("backward",36,foldArm,1)

predicted_class = 0
print("Push out sort ramp predicted_class")
if predicted_class == 0:
  #this belongs to dad - push out the ramp
  moveESCMotor("forward",26,sortingRamp,1)

time.sleep(2)
print("drop onto sorting ramp")
move4Motors(foldHorizontal1,180,150,foldHorizontal2,0,30,foldHorizontal3,180,150,foldHorizontal4,0,30) #wiggle
move4Motors(foldHorizontal1,180,150,foldHorizontal2,0,30,foldHorizontal3,180,150,foldHorizontal4,0,30) #wiggle
move4Motors(foldHorizontal1,180,0,foldHorizontal2,0,180,foldHorizontal3,180,0,foldHorizontal4,0,180) #optn
time.sleep(0.5) #time for clothes to drop

print("check that garment fell correctly. Press enter to prep arms for next garment")
#Prep for next garment - assembly out 50 seconds. pull back ramp if needed
if predicted_class == 0:
  #this was dad clothes, pull back ramp
  moveESCMotor("backward",28,sortingRamp,1)
moveESCMotor("forward",50,armInOut,1) #push out assembly


