import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import face_recognition
import os
from PIL import Image, ImageTk
import threading
import time

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition GUI")
        self.root.geometry("1024x768")  # Larger window size
        
        # Video frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # Buttons
        self.start_button = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(pady=10, fill=tk.X)

        self.stop_button = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(pady=10, fill=tk.X)

        self.load_button = tk.Button(self.control_frame, text="Load Known Faces", command=self.load_known_faces)
        self.load_button.pack(pady=10, fill=tk.X)

        self.capture_button = tk.Button(self.control_frame, text="Capture New Face", command=self.capture_new_face)
        self.capture_button.pack(pady=10, fill=tk.X)

        self.toggle_recognition = tk.BooleanVar(value=True)
        self.recognition_check = tk.Checkbutton(
            self.control_frame, 
            text="Enable Recognition", 
            variable=self.toggle_recognition,
            command=self.toggle_face_recognition
        )
        self.recognition_check.pack(pady=10, fill=tk.X)

        # Known faces list
        self.known_list_label = tk.Label(self.control_frame, text="Known Faces:")
        self.known_list_label.pack(pady=(20,5))
        self.known_listbox = tk.Listbox(self.control_frame, height=10)
        self.known_listbox.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.cap = None
        self.running = False
        self.current_frame = None
        self.photo_image = None

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def toggle_face_recognition(self):
        if self.toggle_recognition.get():
            self.update_status("Face recognition enabled")
        else:
            self.update_status("Face recognition disabled")

    def load_known_faces(self):
        folder = filedialog.askdirectory(title="Select Folder with Known Faces")
        if not folder:
            return

        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_listbox.delete(0, tk.END)

        loaded_count = 0
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(folder, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        self.known_listbox.insert(tk.END, name)
                        loaded_count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        self.update_status(f"Loaded {loaded_count} known faces")
        messagebox.showinfo("Info", f"Successfully loaded {loaded_count} faces")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        
        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()

    def stop_camera(self):
        self.running = False
        if self.video_thread.is_alive():
            self.video_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        
        self.video_label.config(image='')
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Camera stopped")

    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Store the current frame for capture_new_face
            self.current_frame = frame.copy()

            # Process frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if self.toggle_recognition.get() and self.known_face_encodings:
                try:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"

                        if True in matches:
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match = face_distances.argmin()
                            if matches[best_match]:
                                name = self.known_face_names[best_match]

                        # Scale up face locations
                        top, right, bottom, left = [v * 2 for v in (top, right, bottom, left)]
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Face recognition error: {e}")

            # Update the display
            self.update_display(frame)
            time.sleep(0.03)  # ~30 FPS

    def update_display(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Clear previous photo image reference
        if hasattr(self, 'photo_image'):
            self.photo_image = None
            
        self.photo_image = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo_image)
        self.video_label.image = self.photo_image  # Keep reference

    def capture_new_face(self):
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            messagebox.showerror("Error", "No frame available")
            return

        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) != 1:
            messagebox.showwarning("Warning", "Make sure exactly one face is visible")
            return

        name = simpledialog.askstring("Name", "Enter the person's name:")
        if not name:
            return

        # Create known_faces directory if it doesn't exist
        os.makedirs("known_faces", exist_ok=True)

        # Crop and save face
        top, right, bottom, left = face_locations[0]
        face_image = rgb_frame[top:bottom, left:right]
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join("known_faces", f"{name}.jpg")
        cv2.imwrite(save_path, face_image_bgr)

        # Reload known faces
        self.load_known_faces()
        messagebox.showinfo("Success", f"Saved face for '{name}'")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
