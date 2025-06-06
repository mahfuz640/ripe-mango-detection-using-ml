import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os # Import os module to check for file existence

# --- Model Definition (from the provided Jupyter Notebook) ---
# Define class names based on the notebook output
class_names = ['OverRipe', 'Ripe', 'UnRipe'] #

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
    transforms.ToTensor(),              #
    transforms.Normalize([0.485, 0.456, 0.406],  #
                         [0.229, 0.224, 0.225])   #
])

# Define the Convolutional Neural Network
class ConvolutionalNetwork(nn.Module): #
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) #
        self.conv2 = nn.Conv2d(6, 16, 3, 1) #
        self.fc1 = nn.Linear(16 * 54 * 54, 120) #
        self.fc2 = nn.Linear(120, 84) #
        self.fc3 = nn.Linear(84, 20) #
        self.fc4 = nn.Linear(20, len(class_names)) #

    def forward(self, X):
        X = F.relu(self.conv1(X)) #
        X = F.max_pool2d(X, 2, 2) #
        X = F.relu(self.conv2(X)) #
        X = F.max_pool2d(X, 2, 2) #
        X = X.view(-1, 16 * 54 * 54) #
        X = F.relu(self.fc1(X)) #
        X = F.relu(self.fc2(X)) #
        X = F.relu(self.fc3(X)) #
        X = self.fc4(X) #
        return F.log_softmax(X, dim=1) #

# --- Tkinter GUI ---
class ImagePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mango Ripeness Predictor")

        self.image_path = None
        self.model = None

        # --- Model Weights File ---
        # IMPORTANT: Replace 'your_model_weights.pth' with the actual name of your .pth file
        # Make sure this file is in the same directory as your script, or provide the full path.
        self.model_weights_path = 'model_weights.pth' # <--- Set your .pth file name here

        # --- Display Model Accuracy from the Notebook/metrics.csv ---
        # The notebook and metrics.csv report a test accuracy of ~97.95%
        self.accuracy_label = tk.Label(root, text="Mango Ripeness Predictor Created by Md. Mahfuzur Rahman", fg="green", font=("Helvetica", 12, "bold"))
        self.accuracy_label.pack(pady=5)

        # --- Load the trained model ---
        try:
            self.model = ConvolutionalNetwork()
            if os.path.exists(self.model_weights_path):
                # Load the state_dict
                self.model.load_state_dict(torch.load(self.model_weights_path, map_location=torch.device('cpu')))
                self.model.eval() # Set the model to evaluation mode
                self.prediction_status_label = tk.Label(root, text=f"AI & Robotics Engineer", fg="blue")
                self.prediction_status_label.pack(pady=5)
            else:
                self.prediction_status_label = tk.Label(root, text=f"Model weights file '{self.model_weights_path}' not found. Prediction will not be accurate.", fg="red")
                self.prediction_status_label.pack(pady=5)
                self.model = None # Ensure model is None if weights are not loaded

        except Exception as e:
            self.prediction_status_label = tk.Label(root, text=f"Error loading model weights: {e}", fg="red")
            self.prediction_status_label.pack(pady=5)
            self.model = None # Ensure model is None if loading fails

        # Create GUI elements
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict Ripeness", command=self.predict_image, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Prediction: N/A", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # Enable predict button initially if model is loaded
        if self.model:
            self.predict_button.config(state=tk.NORMAL)


    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
        )
        if self.image_path:
            self.display_image(self.image_path)
            # Only enable predict button if both image is selected AND model is loaded
            if self.model:
                self.predict_button.config(state=tk.NORMAL)
            else:
                self.predict_button.config(state=tk.DISABLED)
            self.result_label.config(text="Prediction: N/A")
        else:
            self.predict_button.config(state=tk.DISABLED)
            self.result_label.config(text="Prediction: N/A")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS) # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def predict_image(self):
        if self.image_path and self.model:
            try:
                # Load the image using PIL
                image = Image.open(self.image_path).convert("RGB")

                # Apply transformations
                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

                # Move to appropriate device (CPU or GPU)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                input_batch = input_batch.to(device)

                # Perform prediction
                with torch.no_grad():
                    output = self.model(input_batch)
                    probabilities = F.softmax(output, dim=1)
                    predicted_index = torch.argmax(probabilities, dim=1).item()
                    predicted_class = class_names[predicted_index]
                    confidence = probabilities[0][predicted_index].item() * 100

                self.result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)", fg="blue")

            except Exception as e:
                self.result_label.config(text=f"Prediction Error: {e}", fg="red")
        elif not self.model:
            self.result_label.config(text="Prediction: Model not loaded.", fg="red")
        else:
            self.result_label.config(text="Prediction: Please select an image first.", fg="orange")

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.geometry("500x750") # Set initial window size
    root.mainloop()