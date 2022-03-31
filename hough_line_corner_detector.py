import itertools
import math

import cv2
import numpy as np
from sklearn.cluster import KMeans

from preprocessors import Closer
from processors import EdgeDetector


class HoughLineCornerDetector:
    def __init__(self, rho_acc=2, theta_acc=360, thresh=100, output_process=True):
        self.rho_acc = rho_acc
        self.theta_acc = theta_acc
        self.thresh = thresh
        self.output_process = output_process
        self._preprocessor = [
            Closer(output_process=output_process),
            EdgeDetector(output_process=output_process)
        ]

    def __call__(self, image):
        # Step 1: Определение краёв листа
        self._image = image
        for processor in self._preprocessor:
            self._image = processor(self._image)

        # Step 2: Получение листов
        self._lines = self._get_hough_lines(image)

        # Step 3: Получение точек пересечения полученных линий
        self._intersections = self._get_intersections()

        # Step 4: Получение итогового четырёхугольника
        return self._find_quadrilaterals()

    def _get_hough_lines(self, image):
        lines = cv2.HoughLines(
            self._image,
            self.rho_acc,
            np.pi / self.theta_acc,
            self.thresh
        )

        #Вывод промежуточного изображения с прорисованными линиями, полученными в результате работа алгоритма KMeans

        img_p = image.copy()

        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_p, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.imshow("Hough_line", img_p)

        return lines

    def _get_intersections(self):
        #Нахождение пересечений между линиями

        lines = self._lines
        intersections = []
        group_lines = itertools.combinations(range(len(lines)), 2)
        x_in_range = lambda x: 0 <= x <= self._image.shape[1]
        y_in_range = lambda y: 0 <= y <= self._image.shape[0]

        for i, j in group_lines:
            line_i, line_j = lines[i][0], lines[j][0]

            if 80.0 < self._get_angle_between_lines(line_i, line_j) < 100.0:
                int_point = self._intersection(line_i, line_j)

                if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]):
                    intersections.append(int_point)

        return intersections

    def _get_angle_between_lines(self, line_1, line_2):
        rho1, theta1 = line_1
        rho2, theta2 = line_2

        m1 = -(np.cos(theta1) / np.sin(theta1))
        m2 = -(np.cos(theta2) / np.sin(theta2))
        return abs(math.atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)

    def _intersection(self, line1, line2):

        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])

        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    def _find_quadrilaterals(self):
        X = np.array([[point[0][0], point[0][1]] for point in self._intersections])
        kmeans = KMeans(
            n_clusters=4,
            init='k-means++',
            max_iter=100,
            n_init=10,
            random_state=0
        ).fit(X)

        return [[center.tolist()] for center in kmeans.cluster_centers_]