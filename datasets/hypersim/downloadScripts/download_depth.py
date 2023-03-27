import os

for i in range(0, 10):
	os.system('./download.py --directory \'depth\' --contains frame.000'+str(i)+'.depth_meters.hdf5 --silent')

for i in range(10, 14):
	os.system('./download.py --directory \'depth\' --contains frame.00'+str(i)+'.depth_meters.hdf5 --silent')