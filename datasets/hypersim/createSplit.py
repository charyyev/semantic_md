import random
import shutil
import os

#for seggregating folders with different scenes
# foldersList = []

# # Open the file for reading
# with open('dataFolders.txt', 'r') as file:
#     # Loop through each line in the file
#     for line in file:
#         # Print the line to the console
#         foldersList.append(line.replace('\n',''))

# random.shuffle(foldersList)

# with open('valFolders.txt', 'w') as file:
#     for i in range(0,45):
#         file.write(foldersList[i]+str('\n'))

# with open('testFolders.txt', 'w') as file:
#     for i in range(45,90):
#         file.write(foldersList[i]+str('\n'))

# with open('trainFolders.txt', 'w') as file:
#     for i in range(90,len(foldersList)):
#         file.write(foldersList[i]+str('\n'))


##############################################

#for creating validation data and test data folders
#categories = ['image', 'depth', 'semantic']
#for curr in categories:
#    mainSource_path = os.path.join('/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/HyperSim_Data/', curr)
#    mainDest_path = os.path.join('/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/test_Data/', curr)

#    if not os.path.exists(mainDest_path):
#        os.makedirs(mainDest_path)

#    with open('testFolders.txt', 'r') as file:
#        for folder in file:
#            folderName = folder.replace('\n','')
#            source_dir = os.path.join(mainSource_path, folderName)
#            dest_dir = os.path.join(mainDest_path, folderName)
#            shutil.copytree(source_dir, dest_dir)

##################################################

# for creating the data split image path txt files
# val and text path files created using bash command: find <directory> -type f > <fileName.txt>

imgPath = []
valPath = []
testPath = []
trainPath = [] 

with open('val_imgPath.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Print the line to the console
        valPath.append(line.replace('\n',''))

with open('test_imgPath.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Print the line to the console
        testPath.append(line.replace('\n',''))

with open('imagePath.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Print the line to the console
        imgPath.append(line.replace('\n',''))

for path in imgPath:
    if path in valPath or path in testPath:
        continue
    else:
        trainPath.append(path)

with open('train_imgPath.txt', 'w') as file:
    for i in range(0,len(trainPath)):
        file.write(trainPath[i]+str('\n'))
