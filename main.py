import cv2
import numpy as np
import os

class Augmentation:
    def __init__(self, filename):
        self.img = cv2.imread(filename)
        self.filename = filename.split('.')[0]
        self.filetype = filename.split('.')[1]
        self.images = []
        if not os.path.exists(self.filename):
            os.mkdir(self.filename)

    def write_file(self, img, operation):
        path = self.filename
        filename = f'{self.filename}-{operation}.{self.filetype}'
        cv2.imwrite(f'{path}/{filename}', img)
        self.images.append(img)

    def flip(self):
        img = cv2.flip(self.img, 1)
        self.write_file(img, 'flip')

    def low_contrast(self, count=1):
        for i in range(count):
            matrix = np.ones(self.img.shape) * np.random.uniform(0.10,0.90)
            img = np.uint8(cv2.multiply(np.float64(self.img), matrix))
            self.write_file(img, f'low_contrast_{i}')

    def high_contrast(self, count=1):
        for i in range(count):
            matrix = np.ones(self.img.shape) * np.random.uniform(1.10,1.90)
            img = np.uint8(np.clip(cv2.multiply(np.float64(self.img), matrix),0,255))
            self.write_file(img, f'high_contrast_{i}')

    def brighter(self, count=1):
        for i in range(count):
            matrix = np.ones(self.img.shape, dtype='uint8') * np.random.randint(10,90)
            img = cv2.add(self.img, matrix)
            self.write_file(img, f'brighter_{i}')
    
    def darker(self, count=1):
        for i in range(count):
            matrix = np.ones(self.img.shape, dtype='uint8') * np.random.randint(10,90)
            img = cv2.subtract(self.img, matrix)
            self.write_file(img, f'darker{i}')

    def crop(self):
        # New shapes before cropping
        height = self.img.shape[0] * 75 // 100
        width = self.img.shape[1] * 75 // 100

        img = self.img[:height, :width]
        img = cv2.resize(img, (self.img.shape[1],self.img.shape[0]))
        self.write_file(img, 'crop_top_left')

        img = self.img[-height:, :width]
        img = cv2.resize(img, (self.img.shape[1],self.img.shape[0]))
        self.write_file(img, 'crop_bottom_left')

        img = self.img[:height, -width:]
        img = cv2.resize(img, (self.img.shape[1],self.img.shape[0]))
        self.write_file(img, 'crop_top_right')

        img = self.img[-height:, -width:]
        img = cv2.resize(img, (self.img.shape[1],self.img.shape[0]))
        self.write_file(img, 'crop_bottom_right')

        img = self.img[-height:height, -width:width]
        img = cv2.resize(img, (self.img.shape[1],self.img.shape[0]))
        self.write_file(img, 'crop_middle')
        
    def blur(self, count=1):
        for i in range(count):
            kernel = np.random.randint(5,20)
            img = cv2.blur(self.img, (kernel,kernel))
            self.write_file(img, f'blur_{i}')

    def display(self, column):
        # If the number of images is not divisible by column count, generate empty images to avoid errors.
        empty = np.zeros(self.img.shape, np.uint8)
        while len(self.images) % column != 0:
            self.images.append(empty)

        lines = [cv2.hconcat(self.images[i:i+column]) for i in range(0,len(self.images),column)]
        all_lines = cv2.vconcat(lines)
        cv2.imshow('Results', all_lines)
        cv2.waitKey(0)


if __name__ == '__main__':

    # Give the filename as an argument and run the program
    FILENAME = 'snickers.jpg'
    augmentation = Augmentation(FILENAME)

    # Operations
    augmentation.flip()
    augmentation.low_contrast(10)
    augmentation.high_contrast(10)
    augmentation.brighter(10)
    augmentation.darker(10)
    augmentation.crop()
    augmentation.blur(10)

    # Display all the results
    augmentation.display(8)