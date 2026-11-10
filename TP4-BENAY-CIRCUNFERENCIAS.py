import cv2
import numpy as np


def run_hough_circles_motor_visual():
    img = cv2.imread('motor.jpg')
    if img is None:
        print("Error: No se pudo cargar 'motor.jpg'. Asegúrate de que esté en la misma carpeta.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=40,
        minRadius=30,
        maxRadius=120
    )

    result = img.copy()
    detected_center = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))
            radius = int(i[2])
            detected_center = center


            cv2.circle(result, center, 8, (0, 255, 0), -1)
            cv2.circle(result, center, radius, (0, 0, 255), 3)
            cv2.putText(result, f"Centro X0:{center[0]}, Y0:{center[1]}", (center[0] - 80, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"¡Geometría circular detectada con éxito!")
            print(f"Coordenadas del centro 'A' calculadas (X0, Y0): {center}")
            print(f"Radio estimado: {radius}")
            break

    output_filename = 'hough_circles_motor_output.png'
    cv2.imwrite(output_filename, result)
    print(f"Imagen guardada exitosamente como '{output_filename}'")


    cv2.imshow("Transformada Circular de Hough - Centro 'A' del Motor", result)
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hough_circles_motor_visual()