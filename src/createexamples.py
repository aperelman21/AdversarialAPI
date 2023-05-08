import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def normaliza(imagen):
    imagen.reshape(3072)
    min_im = np.min(imagen)
    max_im = np.max(imagen)
    diff = max_im - min_im
    imagen = ((imagen- min_im)*255/diff)
    return imagen.reshape(32,32,3)

test_images = np.load('C:/Users/alexander.perelman/Downloads/x_adversarial_test.npy')
#(training_images_clean, training_labels) , (test_images_clean, test_labels) = cifar10.load_data()
#test_images = test_images.astype(np.uint8)
idx = 8888
imagen = test_images[idx]
imagen == imagen.astype(float) * (255.0/imagen.max())
imagen = imagen.astype(int)
imagen = normaliza(imagen)
print(np.max(imagen))
print(np.min(imagen))
imagen = imagen.astype(np.uint8)
plt.imshow(imagen)
plt.show()

png_image = Image.fromarray(imagen)
png_image.save('test_image4.JPEG')
'''
image2 = Image.open('test_image.JPEGtest_image.JPEG')

data = np.asarray(image2)

if(data.all() == imagen.all()):
    print("iguales")
else:
    print("distintos")

'''

    