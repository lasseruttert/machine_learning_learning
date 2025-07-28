import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Fully connected
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

class DrawingApp:
    def __init__(self, model_path='mnist_model.pth'):
        self.root = tk.Tk()
        self.root.title("Zahlen Erkennung")
        
        # Canvas zum Zeichnen (größer für bessere UX)
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # Buttons
        self.predict_btn = tk.Button(self.root, text="Erkennen", command=self.predict, width=15, height=2)
        self.predict_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.clear_btn = tk.Button(self.root, text="Löschen", command=self.clear_canvas, width=15, height=2)
        self.clear_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Result Label
        self.result_label = tk.Label(self.root, text="Zeichne eine Zahl!", font=("Arial", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Confidence Labels
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Maus-Events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.old_x = None
        self.old_y = None
        
        # Image zum Speichern (weiß auf schwarz wie MNIST)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Model laden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform für Input
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def paint(self, event):
        """Malen auf Canvas"""
        if self.old_x and self.old_y:
            # Auf Canvas zeichnen
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=12, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            # Auf PIL Image zeichnen
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill='black', width=12)
        
        self.old_x = event.x
        self.old_y = event.y
        
    def reset(self, event):
        """Reset der Mausposition"""
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        """Canvas leeren"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Zeichne eine Zahl!")
        
        # Clear confidence labels
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        
    def predict(self):
        """Vorhersage machen"""
        # Image invertieren (MNIST ist weiß auf schwarz)
        img = Image.eval(self.image, lambda x: 255-x)
        
        # Bounding Box finden und zuschneiden
        bbox = img.getbbox()
        if bbox:
            # Mit etwas Padding zuschneiden
            padding = 20
            bbox = (max(0, bbox[0]-padding), max(0, bbox[1]-padding), min(self.canvas_size, bbox[2]+padding), min(self.canvas_size, bbox[3]+padding))
            img = img.crop(bbox)
            
            # Auf quadratisch bringen (wichtig!)
            size = max(img.size)
            new_img = Image.new('L', (size, size), 'black')
            # Zentrieren
            offset = ((size - img.size[0]) // 2, (size - img.size[1]) // 2)
            new_img.paste(img, offset)
            img = new_img
        
        
        # Transform anwenden
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Ergebnis anzeigen
        self.result_label.config(text=f"Erkannte Zahl: {predicted.item()} ({confidence.item()*100:.1f}%)")
        
        # Top 3 Wahrscheinlichkeiten anzeigen
        top3_prob, top3_idx = torch.topk(probabilities[0], 3)
        
        # Clear old confidence labels
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
            
        # Show top 3 predictions
        for i, (prob, idx) in enumerate(zip(top3_prob, top3_idx)):
            label = tk.Label(self.confidence_frame, text=f"{idx.item()}: {prob.item()*100:.1f}%", font=("Arial", 12))
            label.grid(row=0, column=i, padx=10)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # Stelle sicher, dass das Modell gespeichert wurde
    import os
    if not os.path.exists('mnist_model.pth'):
        print("Fehler: mnist_model.pth nicht gefunden!")
        print("Bitte erst das CNN Modell trainieren und speichern.")
    else:
        app = DrawingApp()
        app.run()