import numpy as np
import cv2 as cv

def Rectify(F, pts1, pts2, img1, img2, name):
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    img3 = img1_copy.copy()
    img4 = img2_copy.copy()

    U, S, V = np.linalg.svd(F)
    e1 = V[-1, :]
    e1 = e1 / e1[2]

    U, S, V = np.linalg.svd(F.T)
    e2 = V[-1, :]
    e2 = e2 / e2[2]

    # https://towardsdatascience.com/a-comprehensive-tutorial-on-stereo-geometry-and-stereo-rectification-with-python-7f368b09924a
    h, w, c = img1_copy.shape
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    e2_p = np.matmul(T, e2)
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e2_p = np.matmul(R, e2_p)
    x = e2_p[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [-1 / x, 0, 1]])
    H2 = np.matmul(np.matmul(np.matmul(np.linalg.inv(T), G), R), T)

    e1_p = np.matmul(T, e1)
    e1_p = e1_p / e1_p[2]
    e1x = e1_p[0]
    e1y = e1_p[1]
    if e1x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e1x / np.sqrt(e1x ** 2 + e1y ** 2)
    R2 = a * e1y / np.sqrt(e1x ** 2 + e1y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e1_p = np.matmul(R, e1_p)
    x = e1_p[0]

    G = np.array([[1, 0, 0], [0, 1, 0], [-1 / x, 0, 1]])

    H1 = np.matmul(np.matmul(np.matmul(np.linalg.inv(T), G), R), T)

    #i = 0
    #for x1, y1, x2, y2 in list_kp1_kp2:
    #    x1 = int(pts1[i, 0])
    #    y1 = int(pts1[i, 1])
    #    x2 = int(pts2[i, 0])
    #    y2 = int(pts2[i, 1])
    #    cv.line(img1_copy, (x1, y1), (int(e1[0]), int(e1[1])), (255, 0, 0), 1)
    #    cv.line(img2_copy, (x2, y2), (int(e2[0]), int(e2[1])), (255, 0, 0), 1)
    #    i += 1

    #eplines = cv.hconcat([img1_copy, img2_copy])

    warp1 = cv.warpPerspective(img1_copy, H1, (1200, 1200))
    warp2 = cv.warpPerspective(img2_copy, H2, (1200, 1200))
    warp3 = cv.warpPerspective(img3, H1, (1200, 1200))
    warp4 = cv.warpPerspective(img4, H2, (1200, 1200))
    rectlines = cv.hconcat([warp1, warp2])

    return warp1, warp2, warp3, warp4, rectlines, H1, H2

def multiDepthMap(name,video,K_matrix,frame_skip):
    vid_capture = cv.VideoCapture(video)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        frame_count = vid_capture.get(7)

    i = 0
    toggle = 0
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()    
        if ret == True:
            # frame = cv.rotate(frame,cv.ROTATE_180)
            if i == 0:
                frame1 = frame
                i += 1
                continue

            if i % frame_skip == 0:
                frame2 = frame

                gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

                orb = cv.ORB_create()

                kp1, des1 = orb.detectAndCompute(gray1,None)
                kp2, des2 = orb.detectAndCompute(gray2, None)

                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                matches = bf.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)

                img = cv.drawMatches(gray1,kp1,gray2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # cv.namedWindow('Brute Force ORB Matching', cv.WINDOW_KEEPRATIO)
                # cv.resizeWindow('Brute Force ORB Matching', 1200, 600)
                # cv.imshow('Brute Force ORB Matching', img)
                # cv.waitKey(0)
                
                pts1 = []
                pts2 = []
                
                for j in matches:
                    pts2.append(kp2[j.trainIdx].pt)
                    pts1.append(kp1[j.queryIdx].pt)

                pts1 = np.int32(pts1)
                pts2 = np.int32(pts2)
                
                F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
                pts1 = pts1[mask.ravel()==1]
                pts2 = pts2[mask.ravel()==1]



                E = np.matmul(np.matmul(K_matrix.T, F), K_matrix)
                R1, R2, T = cv.decomposeEssentialMat(E)
                
                warp1, warp2, warp3, warp4, rectlines, H1, H2 = Rectify(F, pts1, pts2, frame1, frame2, name)
                
                stereo = cv.StereoBM_create()
                
                stereo.setNumDisparities(16)
                stereo.setBlockSize(21)
                
                frame1 = frame2

                gray3 = cv.cvtColor(warp3, cv.COLOR_BGR2GRAY)
                gray4 = cv.cvtColor(warp4, cv.COLOR_BGR2GRAY)

                disparity = stereo.compute(gray1,gray2)
                
                disparity = disparity.astype(np.uint8)
                disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)
                
                
                cv.namedWindow('Disparity', cv.WINDOW_KEEPRATIO)
                cv.resizeWindow('Disparity', 400, 800)
                cv.imshow('Disparity', disparity)
                #cv.imshow('warp1', warp1)
                #cv.imshow('warp2', warp2)
                key = cv.waitKey(100)
                if key == ord('q'):
                    break
            
            i += 1
            
        else:
            break


    return

K = np.array([[3.48169175e+03, 0.00000000e+00, 1.30225222e+03],
 [0.00000000e+00, 3.48458454e+03, 1.58133329e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

multiDepthMap("Testudo","IMG_1404.MOV",K,5)

