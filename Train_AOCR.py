import cv2
import os
import shutil
import sys
from pathlib import Path
import optparse

#python3 Train_AOCR.py -d DIR_PATH(Which ends with a '/')

# Give checkpoint path, steps per checkpoints and number of epoch in line 61

#give images max width and height here
dim = (210, 70)

parser = optparse.OptionParser()
parser.add_option('-d', '--dir_path',
    action="store", dest="dirpath",
    help="Enter test image directory", default="Empty")

parser.add_option('-i', '--image',
    action="store", dest="image",
    help="Input image", default="Empty")

options, args = parser.parse_args()


if os.path.exists("/home/username/path/results.csv"):
  os.remove("/home/username/path/results.csv")

p = Path("/home/username/path/results")
if p.is_dir():
	shutil.rmtree('/home/username/path/results')	

if os.path.exists("/home/username/path/annotations-training.txt"):
  os.remove("/home/username/path/annotations-training.txt")

if os.path.exists("/home/username/path/train.tfrecords"):
  os.remove("/home/username/path/train.tfrecords")



f_veh = open('/home/username/path/annotations-training.txt', 'w+')

if options.dirpath != 'Empty':
	for filename in os.listdir(options.dirpath):
	
		name, ext = os.path.splitext(filename)
		name = name.split('_')
		img = cv2.imread(options.dirpath+filename)
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)	
		cv2.imwrite(options.dirpath+filename,img)
		#os.rename(options.dirpath+filename,options.dirpath+temp[0]+ext)
		if ext in ['.png', '.jpg','.jpeg']:
			f_veh.write(options.dirpath+filename+ ' '+name[0]+ '\n')			
				
comm = 'aocr dataset /home/username/path/annotations-training.txt /home/username/path/train.tfrecords'
comm1 = 'aocr train /home/username/path/train.tfrecords --model-dir /home/username/path/checkpoints --max-height 70 --max-width 210 --max-prediction 6 --num-epoch 1000' 
os.system(comm)
os.system(comm1)
