from PyQt5 import QtCore, QtWidgets

class my_UI(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("2023 HW1")
        MainWindow.resize(1050, 550)
        self.mainwidget = QtWidgets.QWidget(MainWindow)
        self.mainwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.mainwidget)

        self.init_ui_groupbox1()
        self.init_ui_groupbox2()
        self.init_ui_groupbox3()
        self.init_ui_groupbox4()
        self.init_ui_groupbox5()
        self.init_ui_groupbox6()

    def init_ui_groupbox1(self): # groupbox 1
        self.groupBox1 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox1.setGeometry(QtCore.QRect(20, 20, 190, 140))
        self.groupBox1.setObjectName("groupBox1")
        self.groupBox1.setTitle("Load Image")

        self.Button_Load_Folder = QtWidgets.QPushButton(self.groupBox1)
        self.Button_Load_Folder.setGeometry(QtCore.QRect(20, 30, 150, 30))
        self.Button_Load_Folder.setObjectName("Button_Load_Folder")
        self.Button_Load_Folder.setText("Load Folder")

        self.Button_Load_Image_L = QtWidgets.QPushButton(self.groupBox1)
        self.Button_Load_Image_L.setGeometry(QtCore.QRect(20, 70, 150, 30))
        self.Button_Load_Image_L.setObjectName("Button_Load_Image_L")
        self.Button_Load_Image_L.setText("Load Image_L")

        self.Button_Load_Image_R = QtWidgets.QPushButton(self.groupBox1)
        self.Button_Load_Image_R.setGeometry(QtCore.QRect(20, 100, 150, 30))
        self.Button_Load_Image_R.setObjectName("Button_Load_Image_R")
        self.Button_Load_Image_R.setText("Load Image_R")

    def init_ui_groupbox2(self): # groupbox 2
        self.groupBox2 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox2.setGeometry(QtCore.QRect(250, 20, 250, 230))
        self.groupBox2.setObjectName("groupBox2")
        self.groupBox2.setTitle("1. Calibration")

        self.Button_1_1 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_1.setGeometry(QtCore.QRect(20, 30, 150, 30))
        self.Button_1_1.setObjectName("Button_1_1")
        self.Button_1_1.setText("1.1 Find corners")
        self.text_1_1 = QtWidgets.QLabel(self.groupBox2)
        self.text_1_1.setGeometry(QtCore.QRect(180, 30, 50, 30))
        self.text_1_1.setObjectName("text_1_1")
        self.text_1_1.setText("0 / 15")

        self.Button_1_2 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_2.setGeometry(QtCore.QRect(20, 70, 150, 30))
        self.Button_1_2.setObjectName("Button_1_2")
        self.Button_1_2.setText("1.2 Find Intrinsic")

        self.Button_1_3 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_3.setGeometry(QtCore.QRect(20, 110, 150, 30))
        self.Button_1_3.setObjectName("Button_1_3")
        self.Button_1_3.setText("1.3 Find Extrinsic")
        self.Spinbox_1_3 = QtWidgets.QSpinBox(self.groupBox2)
        self.Spinbox_1_3.setGeometry(QtCore.QRect(180, 110, 50, 30))
        self.Spinbox_1_3.setObjectName("Spinbox_1_3")

        self.Button_1_4 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_4.setGeometry(QtCore.QRect(20, 150, 150, 30))
        self.Button_1_4.setObjectName("Button_1_2")
        self.Button_1_4.setText("1.4 Find Distortion")

        self.Button_1_5 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_5.setGeometry(QtCore.QRect(20, 190, 150, 30))
        self.Button_1_5.setObjectName("Button_1_5")
        self.Button_1_5.setText("1.5 Show result")
        self.text_1_5 = QtWidgets.QLabel(self.groupBox2)
        self.text_1_5.setGeometry(QtCore.QRect(180, 190, 50, 30))
        self.text_1_5.setObjectName("text_1_5")
        self.text_1_5.setText("0 / 15")

    def init_ui_groupbox3(self): # groupbox 3
        self.groupBox3 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox3.setGeometry(QtCore.QRect(510, 20, 250, 230))
        self.groupBox3.setObjectName("groupBox3")
        self.groupBox3.setTitle("2. Augmented Reality")

        self.text_2_1 = QtWidgets.QLabel(self.groupBox3)
        self.text_2_1.setGeometry(QtCore.QRect(21, 20, 210, 30))
        self.text_2_1.setObjectName("text_2_1")
        self.text_2_1.setText("Input a word less than 6 char")
        self.input_text = QtWidgets.QTextEdit(self.groupBox3)
        self.input_text.setGeometry(QtCore.QRect(20, 45, 210, 30))
        self.input_text.setObjectName("input_text")

        self.Button_2_1 = QtWidgets.QPushButton(self.groupBox3)
        self.Button_2_1.setGeometry(QtCore.QRect(20, 90, 210, 30))
        self.Button_2_1.setObjectName("Button_2_1")
        self.Button_2_1.setText("2.1 Show Words on Board")
        
        self.Button_2_2 = QtWidgets.QPushButton(self.groupBox3)
        self.Button_2_2.setGeometry(QtCore.QRect(20, 130, 210, 30))
        self.Button_2_2.setObjectName("Button_2_2")
        self.Button_2_2.setText("2.2 Show Words Vertically")

    def init_ui_groupbox4(self): # groupbox 4
        self.groupBox4 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox4.setGeometry(QtCore.QRect(770, 20, 250, 230))
        self.groupBox4.setObjectName("groupBox4")
        self.groupBox4.setTitle("3. Stereo Disparity Map")

        self.Button_3_1 = QtWidgets.QPushButton(self.groupBox4)
        self.Button_3_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_3_1.setObjectName("Button_3_1")
        self.Button_3_1.setText("3.1 Stereo Disparity Map")

        self.Button_3_2 = QtWidgets.QPushButton(self.groupBox4)
        self.Button_3_2.setGeometry(QtCore.QRect(20, 60, 210, 30))
        self.Button_3_2.setObjectName("Button_3_2")
        self.Button_3_2.setText("Close images")

    def init_ui_groupbox5(self): # groupbox 5
        self.groupBox5 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox5.setGeometry(QtCore.QRect(250, 260, 250, 250))
        self.groupBox5.setObjectName("groupBox5")
        self.groupBox5.setTitle("4. Sift")

        self.Button_4_1 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_4_1.setObjectName("Button_4_1")
        self.Button_4_1.setText("Load Image 1")

        self.Button_4_2 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_2.setGeometry(QtCore.QRect(20, 70, 210, 30))
        self.Button_4_2.setObjectName("Button_4_2")
        self.Button_4_2.setText("Load Image 2")

        self.Button_4_3 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_3.setGeometry(QtCore.QRect(20, 110, 210, 30))
        self.Button_4_3.setObjectName("Button_4_3")
        self.Button_4_3.setText("4.1 Keypoints")

        self.Button_4_4 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_4.setGeometry(QtCore.QRect(20, 150, 210, 30))
        self.Button_4_4.setObjectName("Button_4_2")
        self.Button_4_4.setText("4.2 Matched Keypoints")

    def init_ui_groupbox6(self): # groupbox 6
        self.groupBox6 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox6.setGeometry(QtCore.QRect(510, 260, 510, 250))
        self.groupBox6.setObjectName("groupBox6")
        self.groupBox6.setTitle("5. VGG19")

        self.Button_5_1 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_5_1.setObjectName("Button_5_1")
        self.Button_5_1.setText("Load Image")

        self.Button_5_2 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_2.setGeometry(QtCore.QRect(20, 70, 210, 30))
        self.Button_5_2.setObjectName("Button_5_2")
        self.Button_5_2.setText("5.1 Show Augmented Images")

        self.Button_5_3 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_3.setGeometry(QtCore.QRect(20, 110, 210, 30))
        self.Button_5_3.setObjectName("Button_5_3")
        self.Button_5_3.setText("5.2 Show Model Structure")

        self.Button_5_4 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_4.setGeometry(QtCore.QRect(20, 150, 210, 30))
        self.Button_5_4.setObjectName("Button_5_2")
        self.Button_5_4.setText("5.3 Show Acc and Loss")

        self.Button_5_5 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_5.setGeometry(QtCore.QRect(20, 190, 210, 30))
        self.Button_5_5.setObjectName("Button_5_5")
        self.Button_5_5.setText("5.4 Inference")

        self.graph5 = QtWidgets.QGraphicsView(self.groupBox6)
        self.graph5.setGeometry(260, 30, 150, 150)
        self.scene = QtWidgets.QGraphicsScene(self.groupBox6)
        self.graph5.setScene(self.scene)

        self.text_predict = QtWidgets.QLabel(self.groupBox6)
        self.text_predict.setGeometry(QtCore.QRect(260, 190, 180, 30))
        self.text_predict.setObjectName("text_predict")
        self.text_predict.setText("Predict = ")