from PyQt5 import QtCore, QtWidgets, QtGui

class GraffitiBoard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(230, 230)
        self.board = QtGui.QImage(self.size(), QtGui.QImage.Format_RGB32)
        self.board.fill(QtCore.Qt.black)
        self.lastPoint = QtCore.QPoint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(self.rect(), self.board, self.board.rect())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            painter = QtGui.QPainter(self.board)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 10, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
        
class my_UI(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("2023 HW2")
        MainWindow.resize(1080, 700)
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
        self.groupBox1.setGeometry(QtCore.QRect(20, 20, 190, 120))
        self.groupBox1.setObjectName("groupBox1")
        self.groupBox1.setTitle("Load")

        self.Button_Load_Image = QtWidgets.QPushButton(self.groupBox1)
        self.Button_Load_Image.setGeometry(QtCore.QRect(20, 30, 150, 30))
        self.Button_Load_Image.setObjectName("Button_Load_Image")
        self.Button_Load_Image.setText("Load Image")

        self.Button_Load_Video = QtWidgets.QPushButton(self.groupBox1)
        self.Button_Load_Video.setGeometry(QtCore.QRect(20, 70, 150, 30))
        self.Button_Load_Video.setObjectName("Button_Load_Video")
        self.Button_Load_Video.setText("Load Video")

    def init_ui_groupbox2(self): # groupbox 2
        self.groupBox2 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox2.setGeometry(QtCore.QRect(250, 20, 250, 80))
        self.groupBox2.setObjectName("groupBox2")
        self.groupBox2.setTitle("1. Background Subtraction")

        self.Button_1_1 = QtWidgets.QPushButton(self.groupBox2)
        self.Button_1_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_1_1.setObjectName("Button_1_1")
        self.Button_1_1.setText("1. Background Subtraction")

    def init_ui_groupbox3(self): # groupbox 3
        self.groupBox3 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox3.setGeometry(QtCore.QRect(250, 110, 250, 120))
        self.groupBox3.setObjectName("groupBox3")
        self.groupBox3.setTitle("2. Optical Flow")

        self.Button_2_1 = QtWidgets.QPushButton(self.groupBox3)
        self.Button_2_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_2_1.setObjectName("Button_2_1")
        self.Button_2_1.setText("2.1 Preprocessing")
        
        self.Button_2_2 = QtWidgets.QPushButton(self.groupBox3)
        self.Button_2_2.setGeometry(QtCore.QRect(20, 70, 210, 30))
        self.Button_2_2.setObjectName("Button_2_2")
        self.Button_2_2.setText("2.2 Video Tracking")

    def init_ui_groupbox4(self): # groupbox 4
        self.groupBox4 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox4.setGeometry(QtCore.QRect(250, 240, 250, 80))
        self.groupBox4.setObjectName("groupBox4")
        self.groupBox4.setTitle("3. PCA")

        self.Button_3_1 = QtWidgets.QPushButton(self.groupBox4)
        self.Button_3_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_3_1.setObjectName("Button_3_1")
        self.Button_3_1.setText("3. Dimension Reduction")

    def init_ui_groupbox5(self): # groupbox 5
        self.groupBox5 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox5.setGeometry(QtCore.QRect(510, 20, 550, 300))
        self.groupBox5.setObjectName("groupBox5")
        self.groupBox5.setTitle("4. MNIST Classifier Using VGG19")

        self.Button_4_1 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_4_1.setObjectName("Button_4_1")
        self.Button_4_1.setText("4.1 Show Model Structure")

        self.Button_4_2 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_2.setGeometry(QtCore.QRect(20, 70, 210, 30))
        self.Button_4_2.setObjectName("Button_4_2")
        self.Button_4_2.setText("4.2 Show Acc and Loss")

        self.Button_4_3 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_3.setGeometry(QtCore.QRect(20, 110, 210, 30))
        self.Button_4_3.setObjectName("Button_4_3")
        self.Button_4_3.setText("4.3 Predict")

        self.Button_4_4 = QtWidgets.QPushButton(self.groupBox5)
        self.Button_4_4.setGeometry(QtCore.QRect(20, 150, 210, 30))
        self.Button_4_4.setObjectName("Button_4_4")
        self.Button_4_4.setText("4.4 Reset")

        self.text_predict_1 = QtWidgets.QLabel(self.groupBox5)
        self.text_predict_1.setGeometry(QtCore.QRect(20, 190, 180, 30))
        self.text_predict_1.setObjectName("text_predict_1")
        self.text_predict_1.setText("Predict = ")

        self.boardbox = QtWidgets.QGroupBox(self.groupBox5)
        self.boardbox.setGeometry(QtCore.QRect(260, 30, 250, 250))
        self.boardbox.setObjectName("boardbox")

        self.graffitiBoard = GraffitiBoard()
        self.boardbox.setLayout(QtWidgets.QVBoxLayout())
        self.boardbox.layout().addWidget(self.graffitiBoard)

    def init_ui_groupbox6(self): # groupbox 6
        self.groupBox6 = QtWidgets.QGroupBox(self.mainwidget)
        self.groupBox6.setGeometry(QtCore.QRect(510, 330, 550, 330))
        self.groupBox6.setObjectName("groupBox6")
        self.groupBox6.setTitle("5. ResNet50")

        self.Button_5_1 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_1.setGeometry(QtCore.QRect(20, 30, 210, 30))
        self.Button_5_1.setObjectName("Button_5_1")
        self.Button_5_1.setText("Load Image")

        self.Button_5_2 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_2.setGeometry(QtCore.QRect(20, 70, 210, 30))
        self.Button_5_2.setObjectName("Button_5_2")
        self.Button_5_2.setText("5.1 Show Images")

        self.Button_5_3 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_3.setGeometry(QtCore.QRect(20, 110, 210, 30))
        self.Button_5_3.setObjectName("Button_5_3")
        self.Button_5_3.setText("5.2 Show Model Structure")

        self.Button_5_4 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_4.setGeometry(QtCore.QRect(20, 150, 210, 30))
        self.Button_5_4.setObjectName("Button_5_2")
        self.Button_5_4.setText("5.3 Show Comparion")

        self.Button_5_5 = QtWidgets.QPushButton(self.groupBox6)
        self.Button_5_5.setGeometry(QtCore.QRect(20, 190, 210, 30))
        self.Button_5_5.setObjectName("Button_5_5")
        self.Button_5_5.setText("5.4 Inference")

        self.graph5 = QtWidgets.QGraphicsView(self.groupBox6)
        self.graph5.setGeometry(260, 30, 250, 250)
        self.scene = QtWidgets.QGraphicsScene(self.groupBox6)
        self.graph5.setScene(self.scene)

        self.text_predict_2 = QtWidgets.QLabel(self.groupBox6)
        self.text_predict_2.setGeometry(QtCore.QRect(350, 280, 180, 30))
        self.text_predict_2.setObjectName("text_predict_2")
        self.text_predict_2.setText("Predict = ")