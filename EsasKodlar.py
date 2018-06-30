# -*- coding: utf-8 -*-

import random
from PyQt4 import QtCore, QtGui
import cv2
from PIL import Image
import numpy as np
from PyQt4.QtGui import *
from tasarim import Ui_Form
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from skimage import io
import os
import sys
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import matplotlib.widgets as widgets
import matplotlib.image as mimg
import cv2
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import color
import matplotlib.patches as mpatches

from skimage.measure import structural_similarity as ssim
from skimage.transform import resize

class MainWindow(QtGui.QMainWindow, Ui_Form):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        self.horSlider.valueChanged.connect(self.blur)      # TAB1 BLUR SLIDER CALLS BLUR FUNCTION WHEN VALUES CHANGES
        self.dial.valueChanged.connect(self.rotate_img)

        self.labelling_btn.clicked.connect(self.labelling)
        self.comboBox_labelling.currentIndexChanged.connect(self.match_from_CB)

        self.kontrol_btn.clicked.connect(self.kontrol)
        self.BlurySlider.valueChanged.connect(self.give_dots)

        self.TM_goruntuler_CB.currentIndexChanged.connect(self.TM_combo)

    @QtCore.pyqtSignature("bool")
    def show_image(self, img_name, width, height):
        pixMap = QtGui.QPixmap(img_name)
        pixMap = pixMap.scaled(width, height)
        pixItem = QtGui.QGraphicsPixmapItem(pixMap)

        scene1 = QGraphicsScene()
        scene1.addItem(pixItem)
        return scene1

    # ------------------BASIC OPERATIONS TAB1----------------------



    #   uploading image for basic operations
    @QtCore.pyqtSignature("bool")
    def on_upload_btn_clicked(self):
        self.fileName = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Temel Islemler icin secin", ".", "Resim dosyalari (*.*)"))

        img = cv2.imread(self.fileName)
        self.upload_H_txt.setText(str(img.shape[0]))
        self.upload_w_txt.setText(str(img.shape[1]))

        w, h = self.uploadGV.width() - 5, self.uploadGV.height() - 5
        self.uploadGV.setScene(self.show_image(self.fileName, w, h))

    #   making blurry image with slider
    def blur(self):

        value = self.horSlider.value()
        self.slider_value_lbl.setText(str(self.horSlider.value()))

        img = cv2.imread(self.fileName, 1)

        avging = cv2.blur(img, (int(value) / 2 + 1, int(value) / 2 + 1))
        cv2.imwrite('./islenen/blur.png', avging)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/blur.png', w, h))

    # rotate image with dial button
    def rotate_img(self):

        val=self.dial.value()
        self.dial_value_lbl.setText(str(val))

        img=Image.open(self.fileName)
        img=img.rotate(int(val)).save("./islenen/rotate.png")

        self.w_r,self.h_r=self.operationGV.width(),self.operationGV.height()
        self.operationGV.setScene(self.show_image("./islenen/rotate.png",self.w_r,self.h_r))

    #   shows pixels color value
    @QtCore.pyqtSignature("bool")
    def on_show_px_value_btn_clicked(self):
        img = cv2.imread(self.fileName)
        x_pix = int(self.H_X_txt.toPlainText())
        y_pix = int(self.W_Y_txt.toPlainText())

        print img[x_pix,y_pix]

        self.pixel_color_txt.setPlainText(str(img[x_pix,y_pix]))

    #   rotate image horizontally
    @QtCore.pyqtSignature("bool")
    def on_h_cevir_btn_clicked(self):
        imag = cv2.imread(self.fileName)
        himg = cv2.flip(imag, 0)
        cv2.imwrite('./islenen/Htest.png', himg)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/Htest.png', w, h))

    #   rotate image vertically
    @QtCore.pyqtSignature("bool")
    def on_v_cevir_btn_clicked(self):
        imag = cv2.imread(self.fileName)
        himg = cv2.flip(imag, 1)
        cv2.imwrite('./islenen/Vtest.png', himg)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/Vtest.png', w, h))

    #   converts image color from BGR to Gray
    @QtCore.pyqtSignature("bool")
    def on_grayScale_btn_clicked(self):
        img = cv2.imread(self.fileName)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./islenen/gray.png', gray_image)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/gray.png', w, h))

    #   binary image
    @QtCore.pyqtSignature("bool")
    def on_binary_btn_clicked(self):
        from PIL import Image

        col = Image.open(self.fileName)
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if x<128 else 255, '1')
        bw.save('./islenen/binary.png')

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/binary.png', w, h))

    #   edge detection
    @QtCore.pyqtSignature("bool")
    def on_getEdge_btn_clicked(self):
        img = cv2.imread(self.fileName)
        edges = cv2.Canny(img, 100, 200)

        cv2.imwrite('./islenen/edge.png', edges)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/edge.png', w, h))

    #   negative image
    @QtCore.pyqtSignature("bool")
    def on_negatif_btn_clicked(self):

        img = cv2.imread(self.fileName, 1)
        img = (255 - img)

        cv2.imwrite('./islenen/negatif.png', img)

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/negatif.png', w, h))
        self.sonuc_fileName = './islenen/negatif.png'

    # resize
    @QtCore.pyqtSignature("bool")
    def on_resize_btn_clicked(self):

        img = cv2.imread(self.fileName)
        dim = (int(self.H_X_txt.toPlainText()), int(self.W_Y_txt.toPlainText()))
        resized = cv2.resize(img, dim)
        cv2.imwrite('./islenen/resized.png', resized)

        img = cv2.imread('./islenen/resized.png')
        self.after_H_txt.setText(str(img.shape[0]))
        self.after_W_txt.setText(str(img.shape[1]))

        w, h = self.operationGV.width() - 5, self.operationGV.height() - 5
        self.operationGV.setScene(self.show_image('./islenen/resized.png', w, h))

    # ------------------BASIC OPERATIONS TAB1 END----------------------



    # ------------------FILTER OPERATIONS TAB---------------------

    @QtCore.pyqtSignature("bool")
    def on_upload_btn_filterTab_clicked(self):
        self.fileName = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_filterTab.width() - 5, self.uploadGV_filterTab.height() - 5
        self.uploadGV_filterTab.setScene(self.show_image(self.fileName, w, h))

    @QtCore.pyqtSignature("bool")
    def on_canny_btn_filter_clicked(self):
        img = cv2.imread(self.fileName)
        canny = cv2.Canny(img, 100, 200)

        cv2.imwrite('./islenen/canny.png', canny)

        w, h = self.operationsGV_filterTab.width() - 5, self.operationsGV_filterTab.height() - 5
        self.operationsGV_filterTab.setScene(self.show_image('./islenen/canny.png', w, h))

    @QtCore.pyqtSignature("bool")
    def on_sobel_btn_filter_clicked(self):
        img = cv2.imread(self.fileName)
        sobe = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

        cv2.imwrite('./islenen/sobel.png', sobe)

        w, h = self.operationsGV_filterTab.width() - 5, self.operationsGV_filterTab.height() - 5
        self.operationsGV_filterTab.setScene(self.show_image('./islenen/sobel.png', w, h))

    @QtCore.pyqtSignature("bool")
    def on_prewitt_btn_filter_clicked(self):
        from scipy.ndimage import prewitt
        img = cv2.imread(self.fileName)
        prewit = prewitt(img)

        cv2.imwrite('./islenen/prewitt.png', prewit)

        w, h = self.operationsGV_filterTab.width() - 5, self.operationsGV_filterTab.height() - 5
        self.operationsGV_filterTab.setScene(self.show_image('./islenen/prewitt.png', w, h))

    # ------------------FILTER OPERATIONS TAB END---------------------

    # ------------------LABELLING OPERATION TAB-----------------------
    @QtCore.pyqtSignature("bool")
    def on_upload_btn_labelling_clicked(self):
        self.fileName_labelling = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Labelling icin dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_labelling.width() - 5, self.uploadGV_labelling.height() - 5
        self.uploadGV_labelling.setScene(self.show_image(self.fileName_labelling, w, h))

    @QtCore.pyqtSignature("bool")
    def labelling(self):
        f_name = self.fileName_labelling
        image = color.rgb2gray(io.imread(f_name))
        cleared = clear_border(image)
        label_image = label(cleared)

        list_e = []

        for i, region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            list_e1 = minr, minc, maxr, maxc
            list_e.append(list_e1)
            bolge = image[minr:maxr,minc:maxc]
            io.imsave('./labelling/Goruntu_'+str(i)+'.png',bolge)

        img = cv2.imread('./resimler/sekiller.png', cv2.IMREAD_COLOR)
        for i in range(len(list_e)):
            cv2.rectangle(img,(list_e[i][1], list_e[i][0]), (list_e[i][3], list_e[i][2]), (137,231,111), 2)
        cv2.imwrite('./islenen/labelling.png', img)
        w, h = self.operationGV_labelling.width()-5,self.operationGV_labelling.height()-5
        self.operationGV_labelling.setScene(self.show_image('./islenen/labelling.png', w, h))
        klasor = os.listdir('./labelling/')
        for item in klasor:
            self.comboBox_labelling.addItem(item)

    @QtCore.pyqtSignature("bool")
    def match_from_CB(self):

        w, h = self.operationGV_labelling_match.width(),self.operationGV_labelling_match.height()
        self.operationGV_labelling_match.setScene(self.show_image("./labelling/"+self.comboBox_labelling.currentText(), w, h))

    @QtCore.pyqtSignature("bool")
    def kontrol(self):
        print("./labelling/" + self.comboBox_labelling.currentText())
        file_n = str("./labelling/" + self.comboBox_labelling.currentText())
        img = cv2.imread(file_n, 1)
        yukseklik = img.shape[0]
        genislik = img.shape[1]
        sekil = ""

        if (self.rbt_dikdort.isChecked()):
            radio = self.rbt_dikdort.text()
        else:
            radio = self.rbt_kare.text()
        for i in range(yukseklik):
            for j in range(genislik):
                if (img[i, j][0] == 255 & img[i, j][1] == 255 & img[i, j][2] == 255):

                    sekil = "Bu goruntu dikdortgen"
                else:
                    sekil = "Bu goruntu " + radio + " degildir"
                    break
        if (sekil == "Bu goruntu dikdortgen"):
            if (yukseklik == genislik):
                if (self.rbt_dikdort.isChecked()):
                    sekil = "Bu goruntu ozel bir Dikdortgen olan Kare "
                else:
                    sekil = "Bu goruntu Kare"

            else:
                if (self.rbt_dikdort.isChecked()):
                    sekil = "Bu goruntu Dikdortgen"
                else:
                    sekil = "Bu goruntu Kare degildir"

        self.kontrol_lbl.setText(sekil)

    # ------------------LABELLING OPERATION END-----------------------


    # ------------------HISTOGRAM OPERATION TAB-----------------------
    @QtCore.pyqtSignature("bool")
    def on_upload_btn_histogram_clicked(self):
        self.fileName_histogram = unicode(
            QtGui.QFileDialog.getOpenFileName(self, u"Histogram esitleme goruntu sec", ".", u"Resim dosyaları (*.*)"))

        w, h = self.uploadGV_histogram.width() - 5, self.uploadGV_histogram.height() - 5
        self.uploadGV_histogram.setScene(self.show_image(self.fileName_histogram, w, h))

    @QtCore.pyqtSignature("bool")
    def on_histogram_treshold_btn_clicked(self):

        img = cv2.imread(self.fileName_histogram)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cv2.imwrite('./islenen/histo_tresh.png', img_output)
        w, h = self.operationGV_histogram.width() - 5, self.operationGV_histogram.height() - 5
        self.operationGV_histogram.setScene(self.show_image('./islenen/histo_tresh.png', w, h))

    @QtCore.pyqtSignature("bool")
    def on_compare_btn_histogram_clicked(self):

        img = cv2.imread(self.fileName_histogram, 0)
        img2 = cv2.imread('./islenen/histo_tresh.png', 0)

        benzerlik_degeri = ssim(img, img2)
        benzerlik_degeri = format(benzerlik_degeri, '.2f')

        img = resize(img, (img.shape[0], img.shape[1]))

        img2 = resize(img2, (img2.shape[0], img2.shape[1]))

        err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
        err = err / float(img.shape[0] * img.shape[1])
        err = format(err, '.2f')


        self.sonuc_lbl.setText("SSIM : " + str(benzerlik_degeri) + "    MSE : " + str(err))

    # ------------------HISTOGRAM OPERATION END-----------------------


    # ------------------BLURIFY IMAGE OPERATION TAB-----------------------

    @QtCore.pyqtSignature("bool")
    def on_upload_btn_blurify_clicked(self):
        self.fileNameBlurify = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_bluarify.width() - 5, self.uploadGV_bluarify.height() - 5
        self.uploadGV_bluarify.setScene(self.show_image(self.fileNameBlurify, w, h))

    def give_dots(self):
        #   write slider value on label
        self.lbl_slider_blur.setText(str(self.BlurySlider.value()))

        #   blurify the image
        m = cv2.imread(self.fileNameBlurify)
        h, w, bpp = np.shape(m)
        yuzde = ((h * w) / 100) * int(self.BlurySlider.value())
        for py in range(0, yuzde):
            m[random.randint(0, h - 1)][random.randint(0, w - 1)] = [0, 0, 0]
        cv2.imwrite('./islenen/gurultu.png', m)
        w, h = self.operationGV_bluarify.width() - 5, self.operationGV_bluarify.height() - 5
        self.operationGV_bluarify.setScene(self.show_image('./islenen/gurultu.png', w, h))
        self.fileNameBlurified = './islenen/gurultu.png'

        #   measuring the similarity
        img = cv2.imread(self.fileNameBlurify, 0)
        img2 = cv2.imread(self.fileNameBlurified, 0)
        benzerlik_degeri = ssim(img, img2)
        benzerlik_degeri = format(benzerlik_degeri,'.2f')

        img = resize(img, (img.shape[0], img.shape[1]))
        img2 = resize(img2, (img2.shape[0], img2.shape[1]))

        err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
        err = err / float(img.shape[0] * img.shape[1])
        err = format(err,'.2f')
        self.sonuc_lbl_blurify.setText(" SSIM : " + str(benzerlik_degeri)+"  MSE : " + str(err))

    # ------------------  BLURIFY IMAGE OPERATION END -----------------------


    # --------------------  LOGICAL OPERATIONS TAB  ------------------------------

    @QtCore.pyqtSignature("bool")
    def on_upload_btn_logical_clicked(self):
        self.fileName = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_logical.width() - 5, self.uploadGV_logical.height() - 5
        self.uploadGV_logical.setScene(self.show_image(self.fileName, w, h))

    @QtCore.pyqtSignature("bool")
    def on_upload_btn_logical_2_clicked(self):
        self.fileName2 = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_logical_2.width() - 5, self.uploadGV_logical_2.height() - 5
        self.uploadGV_logical_2.setScene(self.show_image(self.fileName2, w, h))
    @QtCore.pyqtSignature("bool")
    def on_operate_btn_logical_clicked(self):
        img1 = cv2.imread(self.fileName)
        img2 = cv2.imread(self.fileName2)
        print img1

        if(self.CB_logical.currentText()=="AND"):
            and_out = cv2.bitwise_and(img1, img2)
            cv2.imwrite('./islenen/and_out.png', and_out)

            w, h = self.operationGV_logical.width() - 5, self.operationGV_logical.height() - 5
            self.operationGV_logical.setScene(self.show_image('./islenen/and_out.png', w, h))

        elif(self.CB_logical.currentText()=="OR"):
            or_out=cv2.bitwise_or(img1, img2)
            cv2.imwrite('./islenen/or_out.png', or_out)

            w, h = self.operationGV_logical.width() - 5, self.operationGV_logical.height() - 5
            self.operationGV_logical.setScene(self.show_image('./islenen/or_out.png', w, h))

        elif(self.CB_logical.currentText()=="XOR"):
            xor_out = cv2.bitwise_xor(img1, img2)
            cv2.imwrite('./islenen/xor_out.png', xor_out)

            w, h = self.operationGV_logical.width() - 5, self.operationGV_logical.height() - 5
            self.operationGV_logical.setScene(self.show_image('./islenen/xor_out.png', w, h))

        elif(self.CB_logical.currentText()=="NOT"):
            not_out = cv2.bitwise_not(img1)
            cv2.imwrite('./islenen/not_out.png', not_out)

            w, h = self.operationGV_logical.width() - 5, self.operationGV_logical.height() - 5
            self.operationGV_logical.setScene(self.show_image('./islenen/not_out.png', w, h))

    # --------------------LOGICAL OPERATIONS END------------------------------



    # --------------------EROSION DILATION OPERATIONS TAB------------------------------

    @QtCore.pyqtSignature("bool")
    def on_upload_btn_ErDil_clicked(self):
        self.fileName = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        w, h = self.uploadGV_ErDil.width() - 5, self.uploadGV_ErDil.height() - 5
        self.uploadGV_ErDil.setScene(self.show_image(self.fileName, w, h))

    @QtCore.pyqtSignature("bool")
    def on_erode_btn_ErDil_clicked(self):
        img = cv2.imread(self.fileName)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)

        cv2.imwrite('./islenen/erosion.png', erosion)

        w, h = self.operationGV_ErDil.width() - 5, self.operationGV_ErDil.height() - 5
        self.operationGV_ErDil.setScene(self.show_image('./islenen/erosion.png', w, h))
        self.fileName='./islenen/erosion.png'

    @QtCore.pyqtSignature("bool")
    def on_dilation_btn_ErDil_clicked(self):
        img = cv2.imread(self.fileName)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)

        cv2.imwrite('./islenen/erosion.png', dilation)

        w, h = self.operationGV_ErDil.width() - 5, self.operationGV_ErDil.height() - 5
        self.operationGV_ErDil.setScene(self.show_image('./islenen/erosion.png', w, h))
        self.fileName = './islenen/erosion.png'

    # --------------------EROSION DILATION OPERATIONS END------------------------------

    # --------------------TEMPLATE MATCHING TAB------------------------------

    @QtCore.pyqtSignature("bool")
    def on_TM_upload_btn_clicked(self):
        self.fileNameTM = unicode(
            QtGui.QFileDialog.getOpenFileName(self, "Goruntuyu sec", ".", "Resim dosyalari (*.*)"))
        f_name = self.fileNameTM

        image = color.rgb2gray(io.imread(f_name))
        cleared = clear_border(image)
        label_image = label(cleared)

        liste = []

        for i, region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            liste1 = minr, minc, maxr, maxc
            liste.append(liste1)
            bolge = image[minr:maxr, minc:maxc]
            io.imsave('./template/image_' + str(i) + '.png', bolge)
        img = cv2.imread('./resimler/sekiller.png', cv2.IMREAD_COLOR)

        for i in range(len(liste)):
            cv2.rectangle(img, (liste[i][1], liste[i][0]), (liste[i][3], liste[i][2]), (137, 231, 111), 2)

        cv2.imwrite('./islenen/labeling.png', img)
        combo = os.listdir('./template/')
        for item in combo:
            self.TM_goruntuler_CB.addItem(item)

        w, h = self.TM_upload_GV.width() - 5, self.TM_upload_GV.height() - 5
        self.TM_upload_GV.setScene(self.show_image('./islenen/labeling.png', w, h))


    def TM_combo(self):
        w,h=self.TM_arananSekil_GV.width()-5,self.TM_arananSekil_GV.height()-5
        self.TM_arananSekil_GV.setScene(self.show_image("./template/"+self.TM_goruntuler_CB.currentText(),w,h))


    @QtCore.pyqtSignature("bool")
    def on_TM_tempMatch_btn_clicked(self):
        img = cv2.imread(self.fileNameTM, 0)
        img2 = img.copy()
        template = cv2.imread("./template/" + str(self.TM_goruntuler_CB.currentText()), 0)
        w, h = template.shape[::-1]
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF_NORMED']
        i = 0
        print len(methods)
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            print i
            # Apply template Matching
            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            print top_left, bottom_right
            cv2.rectangle(img, top_left, bottom_right, 128, 2)
            plt.subplot(121), plt.imshow(res, cmap='gray')
            i = i + 1
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.savefig('./TemplateMatching/sonuc' + str(i) + '.png')
            plt.show()

            img1 = Image.open('./TemplateMatching/sonuc' + str(i) + '.png')
            img12 = img1.crop((52, 63, 208, 218))
            img12.save('./TemplateMatching/temp/' + str(meth) + '.png')

            ima = cv2.imread(self.fileNameTM)
            cv2.rectangle(ima, top_left, bottom_right, (0, 0, 255), 2)
            cv2.imwrite('./TemplateMatching/bolge/' + str(meth) + '.png', ima)

            img11 = Image.open(self.fileNameTM)
            img12 = img11.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
            img12.save('./TemplateMatching/sonuc/' + str(meth) + '.png')

        w, h = self.TM_1_1_GV.width() - 5, self.TM_1_1_GV.height() - 5
        self.TM_1_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCOEFF.png", w, h))
        self.TM_2_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_SQDIFF.png", w, h))
        self.TM_3_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_SQDIFF_NORMED.png", w, h))
        self.TM_4_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCOEFF_NORMED.png", w, h))
        self.TM_5_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCORR_NORMED.png", w, h))
        self.TM_6_1_GV.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCORR.png", w, h))

        self.TM_1_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCOEFF.png", w, h))
        self.TM_2_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_SQDIFF.png", w, h))
        self.TM_3_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_SQDIFF_NORMED.png", w, h))
        self.TM_4_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCOEFF_NORMED.png", w, h))
        self.TM_5_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCORR_NORMED.png", w, h))
        self.TM_6_2_GV.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCORR.png", w, h))

        self.TM_1_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCOEFF.png", w, h))
        self.TM_2_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_SQDIFF.png", w, h))
        self.TM_3_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_SQDIFF_NORMED.png", w, h))
        self.TM_4_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCOEFF_NORMED.png", w, h))
        self.TM_5_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCORR_NORMED.png", w, h))
        self.TM_6_3_GV.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCORR.png", w, h))

        listem = []
        img = cv2.imread("./template/" + str(self.TM_goruntuler_CB.currentText()), 0)
        sonuc = os.listdir('./TemplateMatching/sonuc/')

        for item in sonuc:
            print ('./TemplateMatching/sonuc/' + item)
            img2 = cv2.imread('./TemplateMatching/sonuc/' + item, 0)

            benzerlik_degeri = ssim(img, img2)
            benzerlik_degeri = format(benzerlik_degeri, ".2f")

            err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
            err = err / float(img.shape[0] * img.shape[1])
            err = format(err, ".2f")

            listem.append([benzerlik_degeri, err])

        self.TM_bnzr1_lbl.setText("SSIM:" + str(listem[0][0]) + " MSE:" + str(listem[0][1]))
        self.TM_bnzr4_lbl.setText("SSIM:" + str(listem[1][0]) + " MSE:" + str(listem[1][1]))
        self.TM_bnzr6_lbl.setText("SSIM:" + str(listem[2][0]) + " MSE:" + str(listem[2][1]))
        self.TM_bnzr5_lbl.setText("SSIM:" + str(listem[3][0]) + " MSE:" + str(listem[3][1]))
        self.TM_bnzr2_lbl.setText("SSIM:" + str(listem[4][0]) + " MSE:" + str(listem[4][1]))
        self.TM_bnzr3_lbl.setText("SSIM:" + str(listem[5][0]) + " MSE:" + str(listem[5][1]))


