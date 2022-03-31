import cv2

#Класс определения вершин листка, по которым впоследствии будет проводиться обрезка

class EdgeDetector:
    def __init__(self, output_process = False):
        self.output_process = output_process


    def __call__(self, image, thresh1 = 50, thresh2 = 150, apertureSize = 3):
        edges = cv2.Canny(image, thresh1, thresh2, apertureSize = apertureSize)
        if self.output_process: cv2.imwrite('output/edges.jpg', edges)
        return edges