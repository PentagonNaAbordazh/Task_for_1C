import cv2
import numpy as np

#Основной файл, в котором определяется класс обрезки изображения, а также происходит загрузка исходника.

class PageExtractor:
    def __init__(self, preprocessors, corner_detector, output_process = False):
        assert isinstance(preprocessors, list), "List of preprocessors expected"
        self._preprocessors = preprocessors
        self._corner_detector = corner_detector
        self.output_process = output_process


    def __call__(self, image_path):
        # Step 1: Чтение изображения из файла
        self._image = cv2.imread(image_path)

        # Step 2: Препроцессинг (преобразование изображения для дальнейшего выделения границ)
        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)

        # Step 3: Нахождение точек, по которым будет проводиться обрезка изображения
        self._intersections = self._corner_detector(self._processed)

        # Step 4: Обрезка изображения
        return self._extract_page()

    def _extract_page(self):
        # Получение точек и обрезка изображения
        pts = np.array([
            (x, y)
            for intersection in self._intersections
            for x, y in intersection
        ])
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],  # Вехняя левая точка
            [maxWidth - 1, 0],  # Верхняя правая точка
            [maxWidth - 1, maxHeight - 1],  # Нижняя правая точка
            [0, maxHeight - 1]],  # Нижняя левая точка
            dtype="float32"
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self._processed, M, (maxWidth, maxHeight))

        if self.output_process: cv2.imwrite('output/deskewed.jpg', warped)

        return warped

    def _order_points(self, pts):
        # Инициализация списка координат:
        # 1ая точка -> верхняя левая
        # 2ая точка -> верхняя правая
        # 3яя точка -> нижняя правая
        # 4ая точка -> нижняя левая
        rect = np.zeros((4, 2), dtype="float32")

        # нижняя левая точка будет иметь наименьшую сумму, верхняя правая - наибольшую
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Возвращаем координаты
        return rect

if __name__ == "__main__":
    from hough_line_corner_detector import HoughLineCornerDetector
    from preprocessors import Resizer, OtsuThresholder, FastDenoiser

    #Загружаем данные в функцию обрезания
    page_extractor = PageExtractor(
        preprocessors = [
            Resizer(height = 1280, output_process = True),
            FastDenoiser(strength = 9, output_process = True),
            OtsuThresholder(output_process = True)
        ],
        corner_detector = HoughLineCornerDetector(
            rho_acc=1,
            theta_acc=180,
            thresh=100,
            output_process = True
        )
    )
    # Показываем получившиеся изображения
    extracted = page_extractor('List1.jpg')
    cv2.imwrite("output/output.jpg", extracted)
    cv2.imshow("Extracted page", extracted)
    cv2.waitKey(0)