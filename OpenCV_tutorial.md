# OpenCV

### Привести изображение к ЧБ

```py
import cv2

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### Углы изображения

`Второй и третий параметр` - точности определения углов (уточнить), большая точность => меньше значение

`cv2.Canny(...)` нормально работает дс черными и цветными изображениями

```py
import cv2

img = cv2.Canny(img, 90, 120)
```

Про пороги:
- `первый`: все цвета с порогом **<90** будут считаться черным цветом - 0 (не объект)
- `второй`: все цвета **>140** будут сделаны как белый цвет - 255
- `между`: 

#### Изменение обводки

`kernel` - ядро обводки
`iterations` - кол-вл повторений

По сути это дилатация

```py
import cv2
import numpy as np

# матрица 5 на 5, тип данных - целое число
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
```

По сути эрозия

```py
import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)
img = cv2.erode(img, kernel, iterations=1)
```


## Создание своего изображения

в `(300, 300, 3)` лучше указать еще тройку, чтобы работать с цветами

```py
import cv2
import numpy as np

photo = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imshow("Photo", photo)
```

- `photo[:] = 255, 0, 0` - покрасить картинку
- `cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 255), thickness=3)` - нарисовать квадрат (img, (start point), (h, w), (BGR обводка), толщина обводки в px)
- `cv2.line(img, (0,0), (10, 10), (255, 255, 255), thickness=3)` - линия (img, (start point), (end point), (BGR color), thickness)
- `cv2.circle(img, center, radius, color, thickness)`
- `cv2.putText(photo, "text", (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (BGR color), 3)` - кортеж смещения текста, `cv2.FONT_HERSHEY_TRIPLEX` - это шрифт, 1 - это размер, 3 - это размер обводки текста


## Контуры изображения 

1. Привести картинку к серому: `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
2. Немного размыть, чтобы сгладить углы: `cv2.GaussianBlur(...)` - указываем только нечетные числа
3. Ищем края: `cv2.Canny(img, 100, 100)`
4. Ищем контуры:

```py
import cv2

contours, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
```
- `contours` - список со всем позициями контуров
- `hir` - иерархия объектов
- `img` - то изображение, где уже есть края (после обработки `Canny()`)
- `второй параметр` - режим получения контуров (в данном случае получаем полностью все доступные контуры)
- `третий параметр` - метод получения контуров (в данном случае находит все координаты всех контуров) `cv2.CHAIN_APPROX_SIMPLLE` - более оптимизированный и находи начало и конец контуров
- `` - 
 5. Выводим изображение: `cv2.imshow("...", con)`
 6. Нарисовать контуры:

```py
import cv2

cv2.drawContours(img, contours, -1, (BGR color), thickness=1)
cv2.imshow("...", img)
 ```

- `-1` - это id контуров


# Начальная конфигурация

```py
import cv2


capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
```


## Drawing

**Функции**: cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText()

```python
import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness
```

## Acessing Pixels

```python
import cv2

img = cv2.imread('messi5.jpg')
px = img[100,100]
```


## Image properties

### Image rows, columns and channels `img.shape`

- (342, 548, 3) - для цветного
- (342, 548) - для ЧБ

### Total number of pixels: `img.size`

### Image datatype: `img.dtype`

### Channels

```
b, g, r = cv2.split(img)
img = cv2.merge((b,g,r))
```

## Filter Noises with Blurring

### Averaging 

Берет ядро (область) вокруг пикселя, высчитывает среднее для этой области, 
заменяет интенсивность пикселя на это среднее

```
img = cv2.imread('opencv_logo.png')
blur = cv2.blur(img, (5, 5))
```

### Gaussian Filter

Убирает гауссовские шумы. Задаем ядро `(5, 5)` для фильтра (положительное и желательно нечетное).
`0` - стандартное отклонение, может быть задано как X и Y 
```
blur = cv2.GaussianBlur(img,(5,5),0)
```

### Median Filter

Хорошо убирвает соль и перец.

```
median = cv2.medianBlur(img,5)
```


### Bilateral Filtering

Размытие с сохранением краев

```
blur = cv2.bilateralFilter(img,9,75,75)
```






