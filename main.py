from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Učitaj trenirani model (možeš zamijeniti sa 'last.pt')
model = YOLO("trenirani modeli/100epoha/best.pt")  # npr. 'runs/detect/train/weights/best.pt'

folder_path= "TestData/Slike"  # Putanja do mape sa slikama
# Lista slika koje želiš testirati
slike = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Prođi kroz slike i prikaži rezultate
for img_name in slike:
    img_path = os.path.join(folder_path, img_name)
    results = model(img_path, save=True)  # automatski snima u runs/detect/predict/
    
    # Opcionalno: prikaži rezultat i u Python prozoru
    img_with_boxes = results[0].plot()  # dobiješ NumPy sliku sa anotacijama
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f"Rezultat: {img_path}")
    plt.axis('off')
    plt.show()
