import cv2
import numpy as np
import cudasift
import time

debugFlag = True

def printd(*args):
    if debugFlag:
        print(*args)

class ImageStitching():
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        # self.sift=cv2.xfeatures2d.SIFT_create()
        # self.sift = cv2.ORB_create(nfeatures=1000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.smoothing_window_size = 100
        self.img1 = None
        self.img2 = None
        self.shareview = None
        self.data1 = cudasift.PySiftData()
        self.data2 = cudasift.PySiftData()
        self.last_Hmatrix = np.ndarray(shape=(3, 3), dtype=float, order='F')

    def registration_gpu_old(self, img1, img2):
        t0 = time.time()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data1 = cudasift.PySiftData()
        cudasift.ExtractKeypoints(img1, data1)
        data2 = cudasift.PySiftData()
        cudasift.ExtractKeypoints(img2, data2)
        printd("length:", len(data1), len(data2))
        cudasift.PyMatchSiftData(data1, data2)
        df1, keypoints1 = data1.to_data_frame()


        image1_kp = []
        image2_kp = []

        for index, k in df1.iterrows():
            image1_kp.append((k['xpos'], k['ypos']))
            image2_kp.append((k['match_xpos'], k['match_ypos']))

        image1_kp = np.float32(image1_kp)
        image2_kp = np.float32(image2_kp)

        # ------------compare the goodpoints position to decide the iamge sequence-------
        image1_kp_meanx = np.mean([i[0] for i in image1_kp])
        image2_kp_meanx = np.mean([i[0] for i in image2_kp])
        printd("Image1_kp meanX", image1_kp_meanx)
        printd("Image2_kp meanX", image2_kp_meanx)

        if image1_kp_meanx > image2_kp_meanx:
            seqFlag = 0  # img1|img2
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        else:
            seqFlag = 1  # img2|img1
            H, status = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
        # printd("Finish Regis:", H, seqFlag)
        printd("TIME: ", time.time() - t0)
        return H, seqFlag

    def registration_gpu(self, img1, img2):
        t0 = time.time()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        printd("convert-TIME: ", time.time() - t0)
        # data1 = cudasift.PySiftData()
        cudasift.ExtractKeypoints(img1, self.data1)
        # printd("CUDA-1-prematch: ", data1.to_data_frame())
        # data2 = cudasift.PySiftData()
        cudasift.ExtractKeypoints(img2, self.data2)
        printd("length:", len(self.data1), len(self.data2))
        cudasift.PyMatchSiftData(self.data1, self.data2)
        df1, keypoints1 = self.data1.to_data_frame()
        df1 = df1[(df1.score > 0.8) & (df1.ambiguity < 0.95)]

        # printd("extract-GPU-TIME: ", time.time() - t0)
        t0 = time.time()

        image1_kp = []
        image2_kp = []

        for index, k in df1.iterrows():
            image1_kp.append((k['xpos'], k['ypos']))
            image2_kp.append((k['match_xpos'], k['match_ypos']))


        # Decide transmit shareview or single view
        printd("Matched Length: ", len(image1_kp))
        if len(image1_kp) < 70:
            return None, None
        # printd("formatkp-TIME: ", time.time() - t0)
        t0 = time.time()

        image1_kp = np.float32(image1_kp)
        image2_kp = np.float32(image2_kp)

        # printd("formatkp-float32-TIME: ", time.time() - t0)
        t0 = time.time()

        # ------------compare the goodpoints position to decide the iamge sequence-------
        image1_kp_meanx = np.mean([i[0] for i in image1_kp])
        image2_kp_meanx = np.mean([i[0] for i in image2_kp])
        # ------------Pandas mean is slower----------------------
        # df1_mean = df1.mean()
        # image1_kp_meanx = df1_mean['xpos']
        # image2_kp_meanx = df1_mean['match_xpos']
        printd("Image1_kp meanX: ", image1_kp_meanx)
        printd("Image2_kp meanX: ", image2_kp_meanx)

        printd("countmean-TIME: ", time.time() - t0)
        t0 = time.time()

        try:
            if image1_kp_meanx > image2_kp_meanx:
                seqFlag = 0  # img1|img2
                H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            else:
                seqFlag = 1  # img2|img1
                H, status = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)

            distance = np.linalg.norm(H - self.last_Hmatrix)
            if distance < 30: # Add the threshold to stabilize the shareview
                H = self.last_Hmatrix
            else:
                self.last_Hmatrix = H
            print("Matrix Distance: ", distance)

            printd("findhomo-TIME: ", time.time() - t0)
            return H, seqFlag
        except:
            return None, None

    def registration(self, img1, img2):
        t0 = time.time()
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []

        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imshow("matchPoint", img3)
        # cv2.imwrite('matching_test.jpg', img3)
        printd("good points: ", len(good_points))
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            # ------------compare the goodpoints position to decide the iamge sequence-------
            image1_kp_meanx = np.mean([i[0] for i in image1_kp])
            image2_kp_meanx = np.mean([i[0] for i in image2_kp])
            printd("Image1_kp meanX", image1_kp_meanx)
            printd("Image2_kp meanX", image2_kp_meanx)
            if image1_kp_meanx > image2_kp_meanx:
                seqFlag = 0  # img1|img2
                H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            else:
                seqFlag = 1  # img2|img1
                H, status = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
            printd("recog-CPU-TIME: ", time.time() - t0)
            return H, seqFlag
        return None, None

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def update(self, seqID):
        # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        if self.img1 is None or self.img2 is None:
            return None

        H, seqFlag = self.registration_gpu(self.img1, self.img2)
        t0 = time.time()
        if H is None:
            return np.hstack((self.img1, self.img2))
            # if seqID == 1:
            #     return self.img1
            # elif seqID == 2:
            #     return self.img2
        if seqFlag: # reverse the image sequence
            img2 = self.img1.copy()
            img1 = self.img2.copy()
        else:
            img1 = self.img1.copy()
            img2 = self.img2.copy()
        printd("shareview BLENDING-reverse-TIME: ", time.time() - t0)
        t0 = time.time()

        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        printd("shareview MASK FINISH-TIME: ", time.time() - t0)
        t0 = time.time()
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        result = panorama1 + panorama2
        self.shareview = result
        printd("shareview wrap results-TIME: ", time.time() - t0)

        try:
            rows, cols = np.where(result[:, :, 0] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 2
            self.shareview = result[min_row:max_row, min_col:max_col, :]
        except:
            printd("shareview crop error: ", min_row, max_row, min_col, max_col)
            pass

        return self.shareview

    def get_share(self):
        return self.shareview

    def gpu_match(self, img1, img2):
        t0 = time.time()
        # printd("Image1 Shape:--------", img1.shape)
        # printd("Image2 Shape:--------", img2.shape)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        printd("convert-TIME: ", time.time() - t0)
        printd("Image1 Shape:--------", img1.shape)
        printd("Image2 Shape:--------", img2.shape)
        if img1.shape[0] < 100 or img2.shape[0] < 100:
            print(img1,img2)
            return None
        cudasift.ExtractKeypoints(img1, self.data1)
        cudasift.ExtractKeypoints(img2, self.data2)
        printd("length:", len(self.data1), len(self.data2))
        cudasift.PyMatchSiftData(self.data1, self.data2)
        df1, keypoints1 = self.data1.to_data_frame()
        # df1 = df1[(df1.score > 0.7) & (df1.ambiguity < 0.95)]
        feature_matched = []
        for index, k in df1.iterrows():
            feature_matched.append([[k['ypos'], k['xpos']],[k['match_ypos'], k['match_xpos']]])
        print("Matched Length: ", len(feature_matched))
        return feature_matched

    def cpu_match(self, img1, img2):
        t0 = time.time()
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []

        feature_matched = []

        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                # good_points.append((m1.trainIdx, m1.queryIdx))
                # good_matches.append([m1])

                feature_matched.append([[kp1[m1.queryIdx].pt[1], kp1[m1.queryIdx].pt[0]],[kp2[m1.trainIdx].pt[1],kp2[m1.trainIdx].pt[0]]])

        return feature_matched


        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imshow("matchPoint", img3)
        # cv2.imwrite('matching_test.jpg', img3)
        printd("good points: ", len(good_points))
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            # ------------compare the goodpoints position to decide the iamge sequence-------
            image1_kp_meanx = np.mean([i[0] for i in image1_kp])
            image2_kp_meanx = np.mean([i[0] for i in image2_kp])
            printd("Image1_kp meanX", image1_kp_meanx)
            printd("Image2_kp meanX", image2_kp_meanx)
            if image1_kp_meanx > image2_kp_meanx:
                seqFlag = 0  # img1|img2
                H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            else:
                seqFlag = 1  # img2|img1
                H, status = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
            printd("recog-CPU-TIME: ", time.time() - t0)
            return H, seqFlag
        return None, None

    def cpu_detect(self, img1):
        print("detectFeature-----------------------------------------------------------------------")
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        return kp1, des1
