# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/usr/lib/pymodules/python2.7")
from PyQt4 import QtGui
from EsasKodlar import MainWindow


def main():
    app = QtGui.QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    return app.exec_()

if __name__ == "__main__":
    main()