

# Todo

1. Перекалибровать камеру:
   
   - Сделать 10 изображений с камеры
   - Получить коэффициенты искажения

2. Убрать искажения (Undistort):
   
   - 
       ```
      # undistort
      dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
      # crop the image
      x,y,w,h = roi
      dst = dst[y:y+h, x:x+w]
      cv2.imwrite('calibresult.png',dst)
       ```

3. 