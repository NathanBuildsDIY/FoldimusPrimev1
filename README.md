# FoldimusPrimev1
DIY AI Laundry Folding Robot. Schematics, 3d printer files, parts list is here.  Youtube video explaining the robot & set up here:

In order to set up your Raspberry Pi, execute these commands (I used a pi 5, but a 4 or zero 2 w should work as well)
```
  python3 -m venv --system-site-packages venv #create a venv - this is needed on latest operating system, you can't change system python packages. Note, it's important to use system packages in venv bc picamera2 is already installed and can't be easily installed otherwise
  source ~/laundry/venv/bin/activate #this activates your venv so you can install python packages in it
  sudo apt remove python3-rpi.gpio
  sudo apt install python3-rpi-lgpio #these 2 are only on pi 5 for servo control. Pi 4 and before should still have gpiozero control via gpio libs
  pip3 install rpi-lgpio 
  pip3 install gpiozero
  pip3 install opencv-python
  sudo apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
  sudo apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
  pip3 install tflite_runtime
```
