import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the model architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 16 * 16, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Image transformation
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict function
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor).item()
        label = "Malignant" if output >= 0.5 else "Benign"
    return label, output, image

# Update GUI
def update_prediction(image_path, model, transform, device, label_var, prob_var, image_label, progress_bar, canvas):
    progress_bar.start()
    root.update()
    label, prob, image = predict_image(image_path, model, transform, device)
    label_var.set(f"Prediction: {label}")
    prob_var.set(f"Probability: {prob:.4f}")
    
    # Display image
    image = image.resize((256, 256))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo
    
    # Update probability bar chart
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(["Benign", "Malignant"], [1 - prob, prob], color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    canvas.figure = fig
    canvas.draw()
    progress_bar.stop()

# Select image
def open_image(label_var, prob_var, image_label, model, transform, device, progress_bar, canvas):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if file_path:
        update_prediction(file_path, model, transform, device, label_var, prob_var, image_label, progress_bar, canvas)

# GUI
root = tk.Tk()
root.title("Skin Cancer Prediction")
root.geometry("500x600")

frame = tk.Frame(root)
frame.pack(pady=10)

image_label = tk.Label(frame, text="No image selected", font=("Arial", 12))
image_label.pack()

progress_bar = ttk.Progressbar(root, mode='indeterminate')
progress_bar.pack(pady=5)

label_var = tk.StringVar(value="Prediction: ")
prob_var = tk.StringVar(value="Probability: ")

tk.Label(root, textvariable=label_var, font=("Arial", 14)).pack()
tk.Label(root, textvariable=prob_var, font=("Arial", 14)).pack()

fig, ax = plt.subplots(figsize=(3, 2))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

button = tk.Button(root, text="Open Image", command=lambda: open_image(label_var, prob_var, image_label, model, val_transform, device, progress_bar, canvas))
button.pack(pady=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(r"D:\courses\huawei NTI AI\last_task\skin_cancer_model.pth", device)

root.mainloop()
