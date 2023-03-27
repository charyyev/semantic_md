import os

# for i in range(0, 10):
# 	os.system('./download.py --directory \'image\' --contains frame.000'+str(i)+'.color.hdf5 --silent')

for i in range(12, 14):
	os.system('./download.py --directory \'image\' --contains frame.00'+str(i)+'.color.hdf5 --silent')