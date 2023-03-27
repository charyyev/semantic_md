import os

# for i in range(0, 10):
# 	os.system('./download.py --directory \'semantic\' --contains frame.000'+str(i)+'.semantic.hdf5 --silent')

for i in range(10, 14):
	os.system('./download.py --directory \'semantic\' --contains frame.00'+str(i)+'.semantic.hdf5 --silent')