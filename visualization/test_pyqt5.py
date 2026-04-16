import sys
from PyQt5.QtWidgets import QApplication, QLabel

if __name__ == "__main__":
    app = QApplication(sys.argv)
    label = QLabel("Hello PyQt5")
    label.setGeometry(0,0,300,200);
    label.show()
    sys.exit(app.exec_())
