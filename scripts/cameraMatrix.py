import numpy as np
import cv2 as cv
import os

# Path where the images are saved
dirPath = './../captures'

# Fetching all files within the path directory
dirList = os.listdir(dirPath)

# Extracting image file name from the list
images = [image for image in dirList if image.lower().endswith('.jpg')]

# Creating a list to save image data
image_data = []
worldPts = []
imgPts = []
checkerSize = (9,6)

for imageData in images:
    image_path = os.path.join(dirPath, imageData)
    # cv.imshow("Preview",cv.imread(image_path))
    # cv.waitKey(1)
    image_data.append(cv.imread(image_path))

i = 1
for image in image_data:
    worldPts1 = []
    imgPts1 = []
    meanErr = 0

    length = image.shape[0]
    width = image.shape[1]

    print(length, "\n", width)

    worldCoor = np.zeros((checkerSize[0] * checkerSize[1], 3), np.float32)
    y, x = np.indices(checkerSize[::-1])
    # print(x*21.5, "\t", y*21.5)
    worldCoor[:, 0] = x.ravel()*21.5    
    worldCoor[:, 1] = y.ravel()*21.5
    # print(worldCoor.shape)
    grayImg = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    status, corners = cv.findChessboardCorners(grayImg, checkerSize)
    # print(corners)

    if status:
        refCorners = cv.cornerSubPix(grayImg, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + 
                                                                          cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        drwCorners = cv.drawChessboardCorners(image, checkerSize, refCorners, status)
        # drwCorners = cv.resize(drwCorners, (800, 600))
        # cv.resize(drwCorners,(int(length*0.125),int(width*0.125)))
        cv.imwrite(f'resultimage{i}.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 90])
        cv.imshow('Checkers image',drwCorners)
        worldPts.append(worldCoor)
        imgPts.append(refCorners)
        worldPts1.append(worldCoor)
        imgPts1.append(refCorners)
        ret, kMat, distCoeff, rotVec, transVec = cv.calibrateCamera(worldPts1, imgPts1, grayImg.shape[::-1], None, None)

        for index in range(len(worldPts1)):
            newImgPts, _ = cv.projectPoints(worldPts1[index],rotVec[index],transVec[index],kMat,distCoeff)
            err = cv.norm(imgPts1[index],newImgPts,cv.NORM_L2)/len(newImgPts)
            meanErr += err

        reprojErr = meanErr/len(worldPts1)
        print(f'\nReprojection Error for image {i}: \n',reprojErr)
        i+=1
        cv.waitKey(0)
        cv.destroyAllWindows()

ret, kMat, distCoeff, rotVec, transVec = cv.calibrateCamera(worldPts, imgPts, grayImg.shape[::-1], None, None)

print(f'\nCalibration / K Matrix: \n', kMat)
