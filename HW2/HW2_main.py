from PyQt5 import QtWidgets
import HW2_window
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = HW2_window.mywindow()
    window.show()
    sys.exit(app.exec_())