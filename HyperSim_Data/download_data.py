import os

for i in range(0, 10):
    os.system(
        "./download.py --directory 'depth' --contains frame.000"
        + str(i)
        + ".depth_meters.hdf5 --silent"
    )
    os.system(
        "./download.py --directory 'image' --contains frame.000"
        + str(i)
        + ".color.hdf5 --silent"
    )
    os.system(
        "./download.py --directory 'semantic' --contains frame.000"
        + str(i)
        + ".semantic.hdf5 --silent"
    )

for i in range(10, 33):
    os.system(
        "./download.py --directory 'depth' --contains frame.00"
        + str(i)
        + ".depth_meters.hdf5 --silent"
    )
    os.system(
        "./download.py --directory 'image' --contains frame.00"
        + str(i)
        + ".color.hdf5 --silent"
    )
    os.system(
        "./download.py --directory 'semantic' --contains frame.00"
        + str(i)
        + ".semantic.hdf5 --silent"
    )
