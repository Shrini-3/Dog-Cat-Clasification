import cv2
test_img = cv2.imread('/content/cat.jpg')
#Dispalys the image
plt.imshow(test_img)
#Dispalys the size
test_img.shape

#reduce the image size
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))

#Predicting the image if array value 1 dog and if array value 0 cat
model.predict(test_input)

import cv2
test_img = cv2.imread('/content/Cute dog.jpg')
plt.imshow(test_img)

#reduce the image size
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))

#Predicting the image if array value 1 dog and if array value 0 cat
model.predict(test_input)
