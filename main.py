import cv2
from hand_tracking import hand_tracking
from face_mesh import face_mesh_tracking
from pose_detection import pose_detection

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przetwarzanie obrazu za pomocą wszystkich trzech funkcji
    frame = hand_tracking(frame)
    frame = face_mesh_tracking(frame)
    frame = pose_detection(frame)

    # Wyświetlanie obrazu
    cv2.imshow('Full Body Tracking', frame)

    # Przerwanie pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
