# PyQT-Image-Processing-Desktop-app

# GÖRÜNTÜ İŞLEME FİNAL SINAVI İÇİN GELİŞTİRİLEN UYGULAMAS

1. TEMEL İŞLEMLER
Uygulamamın ilk kısmında temel işlemler yapılmıştır. Ekran görüntüsü aşağıdaki gibidir.
Bu kısımda yapılan işlemler ise;
- İkiliye, Griye ve Negatife Çevirme
- Kenar Gösterme, Aynalama ve Ters Çevirme
- Pikselleri Gösterme
- Resmi Boyutlandırma
- Resmi Döndürme ve Bulaniklastirma

![1](https://user-images.githubusercontent.com/8350817/42125898-0959f7fc-7c88-11e8-934b-7d61f5761f09.jpg)

1.1)Resim gösterme için Upload button yardımı ile dosyadan seçilen bir resim graphic viewde gösterilmiştir ve resmin boyutu line edite bileşenleri ile yazılmıştır.
1.2) Resmi gri, ikiliye çevirme ve kenarlarıın alması. Gri resmi veya renkli resmi sadece siyah ve beyazdan oluşan renge çevirmektir. Resmi griye çevirme bir resmi alıp onu gri tonlara çevirmekle oluşur.
1.3) Görüntüyü horizonlar Slider bileşeni yardımı ile bulabıklaştırılmıştır. Dial brileşeni yardımı ile ise görüntüyü 360 derece dönderilmiştir.

2. FİLTRELEME İŞLEMLERİ
Filtreleme işlemleri resmin piksel değerlerini değiştirerek değişik filtreler kullanarak resmi değiştirmemizi sağlar. Bu kısımda yaptığımız filtreleme işlemleri;
- CANNY
- SOBEL
- PREWITT
Ekran görüntüsü aşağıdaki gibidir.

![2](https://user-images.githubusercontent.com/8350817/42125899-098127d2-7c88-11e8-90c3-3fe5109cde63.jpg)

Canny filtreleme için Cv2’nin canny değerini kullanılarak yapılır. Değer alınıp self.filename ile graphic viewde gösterilir.

Sobel filtrelemesi için gereken kodda aşağıda görüldüğü gibidir. Cv2nin Gaussian blur yapısı kullanılır, sonra griye çevirme işlemi yapılır ve bu değerler x,y değerlerine aktarılır. Aktarılan değerler convertScaleAbs yapısıyla yeniden x,y değişkenlerine aktarılır ve dstde birleştirilir. Cv2 ile yazdırılır, graphic viewde gösterilir.

Prewitt filtreleme için okuduğumuz resim önce griye çevirme işlemini yapılır. Sonrada kernelx ve kernely değerlerini alıp bunları prewittx ve prewitty değişkenlerine atılır. Prewittx ve prewitty değişkenlerini prewit değişkeninde bireştirip sonç yazdırılır ve graphic viewde gösterilir.  

3. LABELLING
Labelling işleminde resimdeki bir çok şeklin kenarlarını bulup onu croplayıp kare olup olmadığı belirlenir. Ekran görüntüsü aşağıdaki gibidir.

![3](https://user-images.githubusercontent.com/8350817/42125900-09ac219e-7c88-11e8-896e-3e27e6a0f22d.jpg)

İlk önce resim seçilir ve okutulur. Resim griye çevirilir. Region ile şekil sayısı bulunur. Kenar değerleri minr,maxr,minc,maxc gibi değerler ile bulunur. Bulunan şekiller kırmızıyla belirginleştirilir. Bunlar x ve y değişkenine atılır. Bulunan şekillerde combobox değerinin içine atılır. Comboboxtan seçilen item graphic viewde gösterilir. Çıkan sonuca göre kare olup olmadığını belirlenir ve kaç tane şekil olduğu yazdırılır.

4. HİSTOGRAM EŞİTLEME
Histogram eşitleme bir resimdeki renk değerlerinin belli bir yerde kümelenmiş olmasından kaynaklanan renk dağılımı bozukluğunu gidermek için kullanılan bir yöntemdir. 
Bu uygulamada uyarlamalı histogram eşitlemesi kullanılır. Bu resim, "fayans" olarak adlandırılan küçük bloklara ayrılmıştır.Ardından bu blokların her biri, her zamanki gibi histogram eşitlenir. Bu nedenle, küçük bir alanda histogram, küçük bir bölgeye (gürültü olmadıkça) sınırlandırır. 

Gürültü oradaysa güçlenecektir. Bunu önlemek için kontrast sınırlaması yani clahe uygulanır. Herhangi bir histogram kutusu belirtilen kontrast sınırının üzerindeyse, bu pikseller histogram eşitlemesi uygulamadan önce kliplenir ve diğer kutulara eşit olarak dağıtılır. Eşitleme sonrasında, karo sınırlarındaki eserleri kaldırmak için bilinear enterpolasyon uygulanır. MSE ve SSIM değerleri bulunur.

![4](https://user-images.githubusercontent.com/8350817/42125901-09dac09e-7c88-11e8-9dba-a925864aba74.jpg)

5. GÜRÜLTÜ OLUŞTURMA

![5](https://user-images.githubusercontent.com/8350817/42125902-0a09acba-7c88-11e8-8077-4afe2bcf5484.jpg)

Bir resimde gürültü oluşturmayı bir çok yönden yapabiliriz. Resmin gürültü oranını ne kadar arttırırsak orijinal resme benzerliği o kadar azalmış olur. Sliderdan %10-%60 arası değer okutulur. Sp_noise isimli fonksiyon oluşturulur ve  burada resme gürültü eklemek için oluşturulan kodlar eklenir.Resmin 0 ve 1 değerlerine sahip olup olmadığı bakıldıktan sonra bir çıktı verilir. Gürültü olacak resim okutulur ve sp_noise isimli fonksiyondan oluşan çıktı orijinal resim ve sliderdan aldığımız değerle gurultulu resme dönüştürülür. En sonunda da MSE ve SSIM değerleri hesaplanır.

6. MANTIKSAL İŞLEMLER
Mantıksal işlemler and,or,xor,not gibi işlemlerle yapılır. Bu uygulamada NOT,AND,OR,XOR gibi mantıksal işlemler ile resimler oluşturulmuştur.Cv2 bitwise_not, bitwise_or, bitwise_xor, bitwise_and fonksiyonlarıyla yapılır. Uygulamanın ekran görüntüsü ve kod çıktısı aşağıdaki gibidir.

![6](https://user-images.githubusercontent.com/8350817/42125903-0a2d5098-7c88-11e8-9233-5e8e8aefd79b.jpg)

7. EROSION/DILATION
Morfolojik dönüşümler, görüntü şekline dayanan bazı basit işlemlerdir. Erozyonun temel fikri, ön plandaki nesnenin sınırlarını aşındırır. Dilation; erozyonun tam tersi. Burada çekirdeğin en az bir pikseli '1' ise bir piksel öğesi '1' olur. Böylece resimdeki beyaz bölgeyi arttırır veya ön plan nesnesinin boyutu artar. Openingde önce erozyon sonra dilation yapılır. Closingde ise önce dilation sonra erozyon yapılarak gerçekleştirilir. Uygulamanın ekran görüntüsü ve çıktısı aşağıdaki gibidir.

![7](https://user-images.githubusercontent.com/8350817/42125904-0a511ac8-7c88-11e8-8d67-0a9e66d16524.jpg)

8. TEMPLATE MATCHING
Template Matching, daha büyük bir resimdeki bir şablon resmin konumunu bulmak ve aramak için kullanılan bir yöntemdir. OpenCv, bu amaçla cv.matchTemplate() fonksiyonunu kullanır. Orijinal görüntüyü ve methodlarla oluşturulan çıktı görüntüsünü çeşitli karşılaştırma yöntemleriyle karşılaştırır. Uygulamanın ekran görüntüsü ve kod çıktısı aşağıdaki gibidir.

![8](https://user-images.githubusercontent.com/8350817/42125905-0a730854-7c88-11e8-87bc-7ceb30bdb056.jpg)



