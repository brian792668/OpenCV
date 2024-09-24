from PyQt5 import QtWidgets
import HW1_UI as UI
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.models import vgg19_bn
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = UI.my_UI(self)
        self.setup_button()

        # parameters
        self.Q1_objectPoint = np.zeros( (11*8, 3), np.float32 )
        self.Q1_objectPoint[:, :2] = np.mgrid[0:11,0:8].T.reshape(-1, 2)
        self.Q3_img_exist = 1

    def setup_button(self):
        self.ui.Button_Load_Folder.clicked.connect(self.load_folder)
        self.ui.Button_Load_Image_L.clicked.connect(self.load_image_L)
        self.ui.Button_Load_Image_R.clicked.connect(self.load_image_R)
        self.ui.Button_1_1.clicked.connect(self.button_1_1)
        self.ui.Button_1_2.clicked.connect(self.button_1_2)
        self.ui.Button_1_3.clicked.connect(self.button_1_3)
        self.ui.Button_1_4.clicked.connect(self.button_1_4)
        self.ui.Button_1_5.clicked.connect(self.button_1_5)
        self.ui.Button_2_1.clicked.connect(self.button_2_1)
        self.ui.Button_2_2.clicked.connect(self.button_2_2)
        self.ui.Button_3_1.clicked.connect(self.button_3_1)
        self.ui.Button_3_2.clicked.connect(self.button_3_2)
        self.ui.Button_4_1.clicked.connect(self.load_image_1)
        self.ui.Button_4_2.clicked.connect(self.load_image_2)
        self.ui.Button_4_3.clicked.connect(self.button_4_3)
        self.ui.Button_4_4.clicked.connect(self.button_4_4)
        self.ui.Button_5_1.clicked.connect(self.button_5_1)
        self.ui.Button_5_2.clicked.connect(self.button_5_2)
        self.ui.Button_5_3.clicked.connect(self.button_5_3)
        self.ui.Button_5_4.clicked.connect(self.button_5_4)
        self.ui.Button_5_5.clicked.connect(self.button_5_5)

    # groupbox1 ------------------------------------------------------
    def load_folder(self):
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self)    
        print("\nPath = ", self.folder_path)

    def load_image_L(self):
        self.image_L, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.png *.jpg *.jpeg *.bmp)')     
        print("\nFile = ", self.image_L)

    def load_image_R(self):
        self.image_R, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        print("\nFile = ", self.image_R)

    # Q1 groupbox2 ---------------------------------------------------
    def button_1_1(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for i in range(1, 16):
            image = cv2.imread( "{}/{}.bmp".format(self.folder_path, i) )
            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)
                if ret:
                    corners2 = cv2.cornerSubPix(grayimg, corners, (5, 5), (-1, -1), criteria)
                    cv2.drawChessboardCorners(image, (11,8), corners2, ret)
                    resized_image = cv2.resize(image, (600, 500))
                    self.ui.text_1_1.setText("{} / 15".format(i))
                    cv2.imshow("img{}".format(i), resized_image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

    def button_1_2(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objectPoints = []
        imagePoints = []

        for i in range(1, 16):
            image = cv2.imread( "{}/{}.bmp".format(self.folder_path, i) )
            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)
                if ret:
                    objectPoints.append(self.Q1_objectPoint)
                    corners2 = cv2.cornerSubPix(grayimg, corners, (5, 5), (-1, -1), criteria)
                    imagePoints.append(corners2)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objectPoints, imagePoints , grayimg.shape[::-1], None, None)
        print("\ninstrinsic matrix :\n", self.mtx)
        
    def button_1_3(self):
        num = self.ui.Spinbox_1_3.value()
        image = cv2.imread("{}/{}.bmp".format(self.folder_path, num))
        if image is None:                
                print("\nImage not found !!!")
        else:
            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayimg, (11,8))
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(grayimg, corners, (5, 5), (-1, -1), criteria)
                _, rvec, tvec, _ = cv2.solvePnPRansac(self.Q1_objectPoint, corners2, self.mtx, self.dist)
                rotation_mtx, _ = cv2.Rodrigues(rvec)
                extrinsic_mtx = np.hstack([rotation_mtx, tvec])
                print("\n{}.bmp".format(num), "extrinsic matrix :\n",extrinsic_mtx)

    def button_1_4(self):
        print("\nDistortion :\n", self.dist)
    
    def button_1_5(self):
        for i in range(1, 16):
            image = cv2.imread("{}/{}.bmp".format(self.folder_path,i))
            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                undistorted_image = cv2.undistort(grayimg, self.mtx, self.dist)
                self.ui.text_1_5.setText("{} / 15".format(i))
                cv2.namedWindow("two_imgs", 0)
                cv2.imshow("two_imgs", np.hstack([grayimg, undistorted_image]))
                cv2.resizeWindow("two_imgs", 1200, 600)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
        
    # Q2 groupbox3 ---------------------------------------------------
    def button_2_1(self):
        objectPoint = np.zeros((11*8, 3), np.float32)
        objectPoint[:, :2] = np.mgrid[:11, :8].T.reshape(-1, 2)
        objectPoints = []
        img_points = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        text = self.ui.input_text.toPlainText()
        text_pos = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]

        for i in range(1,6):
            image = cv2.imread("{}/{}.bmp".format(self.folder_path,i) )

            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)
                if ret:
                    objectPoints.append(objectPoint)
                    corners2 = cv2.cornerSubPix(grayimg, corners, (11, 11), (-1, -1), criteria)
                    img_points.append(corners2)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objectPoints, img_points, grayimg.shape[::-1], None, None)

        for i in range(1, 6):
            image = cv2.imread("{}/{}.bmp".format(self.folder_path,i) )
            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)

                if ret:
                    corners2 = cv2.cornerSubPix(grayimg, corners, (11, 11), (-1, -1), criteria)
                    _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoint, corners2, self.mtx, self.dist)

                    testfile = cv2.FileStorage("{}/Q2_lib/alphabet_lib_onboard.txt".format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(text)):
                        if text[j] == None:
                            print("\nNo text detected !!")
                        else:
                            w = []
                            wi = []
                            ch = testfile.getNode('{}'.format(text[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+text_pos[j])
                                wi.append(ch[k][1]+text_pos[j])
                            imgpts, _ = cv2.projectPoints( np.float32(w).reshape(-1,3), rvec, tvec, self.mtx, self.dist )
                            imgpts_i, _ = cv2.projectPoints( np.float32(wi).reshape(-1,3), rvec, tvec, self.mtx, self.dist )
                            for a in range(len(ch)):
                                image = cv2.line(image, np.int64(tuple(imgpts[a].ravel())), np.int64(tuple(imgpts_i[a].ravel())), (0, 0, 255), 5)

                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", image)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def button_2_2(self):
        objectPoint = np.zeros((11*8, 3), np.float32)
        objectPoint[:, :2] = np.mgrid[:11, :8].T.reshape(-1, 2)
        objectPoints = []
        img_points = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        text = self.ui.input_text.toPlainText()
        text_pos = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]

        for i in range(1,6):
            image = cv2.imread("{}/{}.bmp".format(self.folder_path,i) )

            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)
                if ret:
                    objectPoints.append(objectPoint)
                    corners2 = cv2.cornerSubPix(grayimg, corners, (11, 11), (-1, -1), criteria)
                    img_points.append(corners2)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objectPoints, img_points, grayimg.shape[::-1], None, None)

        for i in range(1, 6):
            image = cv2.imread("{}/{}.bmp".format(self.folder_path,i) )
            if image is None:                
                print("\nImage not found !!!")
            else:
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayimg, (11,8), None)

                if ret:
                    corners2 = cv2.cornerSubPix(grayimg, corners, (11, 11), (-1, -1), criteria)
                    _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoint, corners2, self.mtx, self.dist)

                    testfile = cv2.FileStorage("{}/Q2_lib/alphabet_lib_vertical.txt".format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(text)):
                        if text[j] == None:
                            print("\nNo text detected !!")
                        else:
                            w = []
                            wi = []
                            ch = testfile.getNode('{}'.format(text[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+text_pos[j])
                                wi.append(ch[k][1]+text_pos[j])
                            imgpts, _ = cv2.projectPoints( np.float32(w).reshape(-1,3), rvec, tvec, self.mtx, self.dist )
                            imgpts_i, _ = cv2.projectPoints( np.float32(wi).reshape(-1,3), rvec, tvec, self.mtx, self.dist )
                            for a in range(len(ch)):
                                image = cv2.line(image, np.int64(tuple(imgpts[a].ravel())), np.int64(tuple(imgpts_i[a].ravel())), (0, 0, 255), 5)

                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", image)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    # Q3 groupbox4 ---------------------------------------------------
    def button_3_1(self):
        image_L = cv2.imread("{}".format(self.image_L))
        image_R = cv2.imread("{}".format(self.image_R))
        grayimg_L = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
        grayimg_R = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create( numDisparities=256, blockSize=25 )
        disparity = stereo.compute( grayimg_L, grayimg_R )

        height, width = disparity.shape[0], disparity.shape[1]
        scale = 0.5
        image_L = cv2.resize(image_L, (int(width*scale),int(height*scale)), interpolation=cv2.INTER_AREA)
        image_R = cv2.resize(image_R, (int(width*scale),int(height*scale)), interpolation=cv2.INTER_AREA)
        disparity = cv2.resize(disparity, (int(width*scale),int(height*scale)), interpolation=cv2.INTER_AREA)

        def dot(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                if disparity[y][x] > 0 :
                    x_right = x - int( (disparity[y][x]) * 0.062*scale )
                    cv2.circle(image_R, (x_right, y), 3, (0, 255, 0), -1)
                    cv2.imshow('image_R', image_R)
                    cv2.moveWindow('image_R', 960, 0)
        
        cv2.namedWindow('image_L')
        cv2.namedWindow('image_R')
        cv2.namedWindow('image_gray')
        cv2.moveWindow('image_L', 10, 0)
        cv2.moveWindow('image_R', 960, 0)
        cv2.moveWindow('image_gray', 500, 500)
        cv2.imshow('image_L', image_L)
        cv2.imshow('image_R', image_R)
        cv2.imshow("image_gray", (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity)))
        while True:
            cv2.setMouseCallback('image_L', dot, None)
            cv2.waitKey(10)
            if self.Q3_img_exist == 0:
                break

    def button_3_2(self):
        cv2.destroyAllWindows()
        self.Q3_img_exist = 0

    # Q4 groupbox5 ---------------------------------------------------
    def load_image_1(self):
        self.image_1, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.jpg *.jpeg *.png)')     
        print("\nFile = ", self.image_1)

    def load_image_2(self):
        self.image_2, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.jpg *.jpeg *.png)')
        print("\nFile = ", self.image_2)

    def button_4_3(self):
        image_1 = cv2.imread("{}".format(self.image_1))
        grayimg = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create() 
        keypoint = sift.detect(grayimg, None)
        image_sift = cv2.drawKeypoints(grayimg, keypoint, None, color=(0, 255, 0))
        resized_grayimg = cv2.resize(grayimg, (500, 500))
        resized_image_sift = cv2.resize(image_sift, (500, 500))

        cv2.namedWindow("left.jpg")
        cv2.namedWindow("Key Point")
        cv2.moveWindow("left.jpg", 10, 0)
        cv2.moveWindow("Key Point", 510, 0)
        cv2.imshow("left.jpg", resized_grayimg)
        cv2.imshow("Key Point", resized_image_sift)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    def button_4_4(self):
        image_1 = cv2.imread("{}".format(self.image_1))
        image_2 = cv2.imread("{}".format(self.image_2))
        grayimg1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        grayimg2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(grayimg1, None)
        kp2, des2 = sift.detectAndCompute(grayimg2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        outImg = np.empty((max(grayimg1.shape[0], grayimg2.shape[0]), grayimg1.shape[1] + grayimg2.shape[1], 3), dtype=np.uint8)
        result_img = cv2.drawMatchesKnn(grayimg1, kp1, grayimg2, kp2, [good], outImg, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        small_result_img = cv2.resize(result_img, (800, 400))

        cv2.namedWindow("Matching Result")
        cv2.imshow("Matching Result", small_result_img)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()

    # Q5 groupbox6 ---------------------------------------------------

    def button_5_1(self):
        from PyQt5.QtGui import QPixmap
        self.img5_1, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.png)')     
        print("\nFile = ", self.img5_1)
        pixmap = QPixmap(self.img5_1)
        pixmap = pixmap.scaled(128, 128)
        self.ui.scene.addPixmap(pixmap)
        self.ui.graph5.setScene(self.ui.scene)
        self.ui.text_predict.setText("Predict = ")
    
    def button_5_2(self):
        folder_path = "Dataset_CvDl_Hw1/Q5_image/Q5_1/"
        T = v2.Compose([v2.RandomRotation(30)])

        # 3x3 subplot
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5)
        plt.tight_layout()

        for i, filename in enumerate(os.listdir(folder_path)[:9]):
            row = i // 3
            col = i % 3
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = T(image)
            axes[row, col].imshow(image)
            axes[row, col].set_title(filename)
            axes[row, col].axis('off')
        plt.show()
              
    def button_5_3(self):
        import torchsummary
        import torchvision.models as models
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.vgg19_bn(num_classes=10)
        model.to(device)
        torchsummary.summary(model, (3, 32, 32))
    
    def button_5_4(self):
        image = None
        image = cv2.imread( "Q5/VGG_batch=8.png" )
        if image is None:
            image = cv2.imread( "HW1/Q5/VGG_batch=8.png" )
        image = cv2.resize(image, (700, 700))
        cv2.imshow("VGG", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    def button_5_5(self):
        model = vgg19_bn(num_classes=10)
        model.load_state_dict(torch.load("Q5/VGG19_cifar_net.pth", map_location=torch.device('cpu')))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_path = self.img5_1
        image = Image.open(image_path)
        image = transform(image)

        with torch.no_grad():
            image = image.unsqueeze(0)
            outputs = model(image)

        # model outputs
        _, predicted = outputs.max(1)
        class_index = predicted.item()
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_class = class_names[class_index]

        # softmax
        import math
        probabilities = outputs[0].tolist()
        total = 0.0
        for i in range(10):
            probabilities[i] = math.exp(probabilities[i])
            total += probabilities[i]
        for i in range(10):
            probabilities[i] = probabilities[i] / total

        plt.bar(class_names, probabilities)
        plt.ylim(0, 1)
        plt.xlabel('Class Names')
        plt.ylabel('Probability')
        plt.title('Predicted Class Probabilities')
        plt.tight_layout()
        plt.show()

        self.ui.text_predict.setText(f"Predict = {predicted_class}")

        return