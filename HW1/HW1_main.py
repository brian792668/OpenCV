from PyQt5 import QtWidgets
import HW1_window
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = HW1_window.mywindow()
    window.show()
    sys.exit(app.exec_())