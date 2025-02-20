import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize device
device = torch.device("cpu")

# Initialize MTCNN (Face Detector) and InceptionResnetV1 (Face Recognition Model)
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# File path for logging entry records
LOG_FILE = "entry_logs.csv"

# Ensure log file exists
def initialize_log():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Name", "Timestamp", "Status"])
        df.to_csv(LOG_FILE, index=False)

initialize_log()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        faces = mtcnn(image)  # Aligned faces
        if faces is None:
            return []

        encodings = []
        for face in faces:
            face_tensor = face.unsqueeze(0).to(device)
            encoding = resnet(face_tensor).detach().cpu().numpy().flatten()
            encodings.append(encoding)
        
        return encodings

# Function to encode known faces
def encode_known_faces(known_faces):
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        image_path = os.path.join(os.getcwd(), image_path)
        known_image = cv2.imread(image_path)
        
        if known_image is not None:
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Define known faces
known_faces = {
    "Kavya": "Images/KavyaSai.jpg",
    "Madhumita": "Images/Madhumita.jpg",
    "Rishitha": "Images/Rishitha.jpg",
    "Sushma": "Images/sushma.jpg"
}

# Encode known faces
known_face_encodings, known_face_names = encode_known_faces(known_faces)

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    
    for test_encoding in test_encodings:
        test_tensor = torch.tensor(test_encoding)
        known_tensors = torch.tensor(known_encodings)

        similarities = F.cosine_similarity(test_tensor.unsqueeze(0), known_tensors)
        max_sim_idx = torch.argmax(similarities).item()
        max_similarity = similarities[max_sim_idx].item()

        if max_similarity > threshold:
            recognized_names.append(known_names[max_sim_idx])
        else:
            recognized_names.append("Not Recognized")

    return recognized_names

# Store logged persons for the current session
logged_persons = set()

def log_entry(name):
    global logged_persons
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log only if the person has NOT been logged in this session
    if name in logged_persons:
        return  # Skip duplicate logging in the same session

    logged_persons.add(name)  # Mark person as logged

    status = "Entry"
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{"Name": name, "Timestamp": timestamp, "Status": status}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    print(f"{name} logged at {timestamp}")

# Start video capture
cap = cv2.VideoCapture(0)
thresh = 0.6
fps = 30
cap.set(cv2.CAP_PROP_FPS, fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured, retrying...")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)
    
    if test_face_encodings and known_face_encodings:
        detection_results = mtcnn.detect(frame_rgb)
        
        if detection_results[0] is not None:
            names = recognize_faces(known_face_encodings, known_face_names, test_face_encodings, thresh)
            
            for name, box in zip(names, detection_results[0]):
                if box is not None:
                    (x1, y1, x2, y2) = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if name != "Not Recognized":
                        log_entry(name)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
