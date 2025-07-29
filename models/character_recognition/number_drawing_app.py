import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
from number_classifier import NumberClassifier



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
        self.model = NumberClassifier().to(self.device)
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