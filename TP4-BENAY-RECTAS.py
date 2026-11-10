import cv2
import numpy as np


def run_hough_lines_motor_visual():
    img = cv2.imread('motor.jpg')
    if img is None:
        print("Error: No se pudo cargar 'motor.jpg'. Asegúrate de que esté en la misma carpeta.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    edges = cv2.Canny(blurred, 100, 200, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    result = img.copy()
    line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] if len(line.shape) > 1 else line
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            line_count += 1

    print(f"Líneas estructurales detectadas en el motor: {line_count}")

    output_filename = 'hough_lines_motor_output.png'
    cv2.imwrite(output_filename, result)
    print(f"Imagen guardada exitosamente como '{output_filename}'")

    cv2.imshow("Transformada de Hough - Deteccion de Rectas en Motor", result)
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hough_lines_motor_visual()