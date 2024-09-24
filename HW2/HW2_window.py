from PyQt5 import QtWidgets, QtCore, QtGui

import HW2_UI as UI
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchsummary
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = UI.my_UI(self)
        self.setup_button()

        # parameters
        # self.Q3_img_exist = 1

    def setup_button(self):
        self.ui.Button_Load_Image.clicked.connect(self.load_Image)
        self.ui.Button_Load_Video.clicked.connect(self.load_Video)

        self.ui.Button_1_1.clicked.connect(self.button_1_1)

        self.ui.Button_2_1.clicked.connect(self.button_2_1)
        self.ui.Button_2_2.clicked.connect(self.button_2_2)

        self.ui.Button_3_1.clicked.connect(self.button_3_1)

        self.ui.Button_4_1.clicked.connect(self.button_4_1)
        self.ui.Button_4_2.clicked.connect(self.button_4_2)
        self.ui.Button_4_3.clicked.connect(self.button_4_3)
        self.ui.Button_4_4.clicked.connect(self.button_4_4)
        self.ui.Button_5_1.clicked.connect(self.button_5_1)
        self.ui.Button_5_2.clicked.connect(self.button_5_2)
        self.ui.Button_5_3.clicked.connect(self.button_5_3)
        self.ui.Button_5_4.clicked.connect(self.button_5_4)
        self.ui.Button_5_5.clicked.connect(self.button_5_5)

    # groupbox1 ------------------------------------------------------
    def load_Image(self):
        self.Image, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.png *.jpg *.jpeg *.bmp)')     
        print("\nFile = ", self.Image)
    def load_Video(self):
        self.Video, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.mp4)')
        print("\nFile = ", self.Video)

    # Q1 groupbox2 ---------------------------------------------------
    def button_1_1(self):
        subtractor = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 700, detectShadows=True)
        cap = cv2.VideoCapture(self.Video)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            mask = subtractor.apply(blurred_frame)
            result_frame = cv2.bitwise_and(frame, frame, mask=mask)

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > 3:
                cv2.imshow("Video Processing", cv2.hconcat([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result_frame]))
                if cv2.waitKey(30) & 0xFF == ord('q') or cv2.getWindowProperty("Video Processing", cv2.WND_PROP_VISIBLE) < 1:
                    break

        cap.release()
        cv2.destroyAllWindows()
        
    # Q2 groupbox3 ---------------------------------------------------
    def resize_image_by_width(image, target_width):
        height, width = image.shape[:2]
        scale_factor = target_width / width
        target_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (target_width, target_height))
        return resized_image
    def button_2_1(self):
        cap = cv2.VideoCapture(self.Video)
        ret, first_frame = cap.read()
        first_frame = cv2.resize(first_frame, (800, int(first_frame.shape[0] * 800 / first_frame.shape[1])))

        gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_first_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

        # 轉換特徵點坐標為整數
        corners = np.int0(corners)

        # 在第一幀中畫上十字標記
        for corner in corners:
            x, y = corner.ravel()
            cv2.line(first_frame, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
            cv2.line(first_frame, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

        # 顯示第一幀
        cv2.imshow('First Frame with Cross Mark', first_frame)
        cv2.waitKey(0)
        cap.release()
    def button_2_2(self):
        cap = cv2.VideoCapture(self.Video)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
        corners = np.float32(corners)   # 轉換特徵點坐標為浮點數
        trajectory = np.array([])    # 創建一個空陣列來存放特徵點的軌跡
        
        while True:
            ret, new_frame = cap.read()
            if not ret:    break
            new_frame = cv2.resize(new_frame, (800, int(new_frame.shape[0] * 800 / new_frame.shape[1])))
            new_gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # cv2.calcOpticalFlowPyrLK追蹤特徵點
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(gray_frame, new_gray_frame, corners, None)
            good_new_corners = new_corners[status == 1]    # 選擇追踪成功的點

            if len(good_new_corners) > 0:
                corners = good_new_corners.reshape(-1, 1, 2)
                gray_frame = new_gray_frame
                trajectory = np.append(trajectory, corners[0])

                if len(trajectory) >= 2:  # 需要至少兩個點來繪製線
                    trajectory = trajectory.reshape(-1, 2)
                    for i in range(1, len(trajectory)):
                        pt1 = (int(trajectory[i - 1][0]), int(trajectory[i - 1][1]))
                        pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                        cv2.line(new_frame, pt1, pt2, (0, 255, 255), 2)
                    cross_pt1 = (int(trajectory[-1][0] - 10), int(trajectory[-1][1]))
                    cross_pt2 = (int(trajectory[-1][0] + 10), int(trajectory[-1][1]))
                    cross_pt3 = (int(trajectory[-1][0]), int(trajectory[-1][1] - 10))
                    cross_pt4 = (int(trajectory[-1][0]), int(trajectory[-1][1] + 10))
                    cv2.line(new_frame, cross_pt1, cross_pt2, (0, 0, 255), 2)
                    cv2.line(new_frame, cross_pt3, cross_pt4, (0, 0, 255), 2)
                    cv2.imshow('Trajectory', new_frame)
                    cv2.waitKey(3)
                else:
                    print("No valid trajectory found.")

        cap.release()
        cv2.destroyAllWindows()

    # Q3 groupbox4 ---------------------------------------------------
    def button_3_1(self):
        from sklearn.decomposition import PCA
        from sklearn.metrics import mean_squared_error
        
        rgb_image = cv2.imread(self.Image)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)      # BGR2RGB
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)    # Convert RGB image to grayscale
        gray_Normalize = gray_image / 255.0   # Normalize grayscale image to [0, 1]

        w, h = gray_Normalize.shape     # Get the dimensions of the image
        n_components = min(w, h)    # Perform PCA for dimension reduction 
        MSE = 0
        n = 0

        while n < n_components:
            n += 1
            pca = PCA(n_components=n)
            transformed_gray = pca.fit_transform(gray_Normalize)
            reconstructed_gray = pca.inverse_transform(transformed_gray).reshape(w, h)
            MSE = mean_squared_error(gray_Normalize, reconstructed_gray) * (255**2)
            print(f"n = {n}  MSE = {round(MSE,1)}")
            if MSE <= 3.0: break

        # Plot the original and reconstructed images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title("Gray scale Image")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_gray, cmap='gray')
        plt.title(f"Reconstructed Image (n={n})")
        plt.axis('off')
        plt.show()

    # Q4 groupbox5 ---------------------------------------------------
    def button_4_1(self):
        model = models.vgg19_bn(num_classes=10)
        model.load_state_dict(torch.load("model/VGG19_MNIST_batch=50.pth", map_location=torch.device('cpu')))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        torchsummary.summary(model, (3, 32, 32))
    def button_4_2(self):
        image = None
        image = cv2.imread( "Q4_train/VGG19_MNIST_batch=300.png" )
        image = cv2.resize(image, (700, 700))
        cv2.imshow("VGG19_accuracy&loss", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    def button_4_3(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        img = self.ui.graffitiBoard.board
        img.save("Q4_train/handwrite.png")
        image = Image.open("Q4_train/handwrite.png")
        image = transform(image)

        model = models.vgg19_bn(num_classes=10)
        model.load_state_dict(torch.load("model/VGG19_MNIST_batch=50.pth", map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            image = image.unsqueeze(0)
            outputs = model(image)

        # model outputs
        _, predicted = outputs.max(1)
        class_index = predicted.item()
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        predicted_class = class_names[class_index]
        self.ui.text_predict_1.setText(f"Predict = {predicted_class}")  

        # softmax
        probabilities = outputs[0].tolist()
        total = 0.0
        for i in range(10):
            probabilities[i] = np.exp(probabilities[i])
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
    def button_4_4(self):
        self.ui.graffitiBoard.board.fill(QtCore.Qt.black)
        self.update()
        self.ui.text_predict_1.setText("Predict = ")

    # Q5 groupbox6 ---------------------------------------------------
    def button_5_1(self):
        from PyQt5.QtGui import QPixmap
        self.img5_1, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter = 'Image Files (*.png), (*.jpg)')     
        print("\nFile = ", self.img5_1)
        pixmap = QPixmap(self.img5_1)
        pixmap = pixmap.scaled(224, 224)
        self.ui.scene.addPixmap(pixmap)
        self.ui.graph5.setScene(self.ui.scene)
        self.ui.text_predict_2.setText("Predict = ")
    def button_5_2(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img1 = Image.open("Dataset_CvDl_Hw2/Q5/Cat/8043.jpg")
        img2 = Image.open("Dataset_CvDl_Hw2/Q5/Dog/12051.jpg")
        resized_image1 = transform(img1)    # PyTorch form (224x224x3) 
        resized_image2 = transform(img2) 
        image_np1 = resized_image1.numpy()
        image_np1 = np.transpose(image_np1, (1, 2, 0)) # to matplotlib form (3x224x224)
        image_np2 = resized_image2.numpy()
        image_np2 = np.transpose(image_np2, (1, 2, 0))

        plt.subplot(1, 2, 1)
        plt.imshow(image_np1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image_np2)
        plt.axis('off')
        plt.show()
    def button_5_3(self):
        import torchsummary
        import torchvision.models as models
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 1),
            torch.nn.Sigmoid()
        )
        model.to(device)
        torchsummary.summary(model, (3, 224, 224))
    def button_5_4(self):
        image = None
        image = cv2.imread( "Q5_train/Accuracy_Comparison.png" )
        # image = cv2.resize(image, (700, 700))
        cv2.imshow("Accuracy_Comparison", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    def button_5_5(self):
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 1),
            torch.nn.Sigmoid()
        )
        model.load_state_dict(torch.load("model/ResNet50_w.pth", map_location=torch.device('cpu')))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(self.img5_1)
        image = transform(image)

        # model outputs
        with torch.no_grad():
            image = image.unsqueeze(0)
            outputs = model(image).item()

        if outputs <= 0.5 :
            self.ui.text_predict_2.setText(f"Predict = Cat")
            print("Cat")
        else :
            self.ui.text_predict_2.setText(f"Predict = Dog")
            print("Dog")