import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
from alphabet_classifier import LetterConvNet

class LetterDrawingApp:
    def __init__(self, model_path='emnist_letters_model.pth'):
        self.root = tk.Tk()
        self.root.title("Buchstaben Erkennung (A-Z)")
        
        # Canvas zum Zeichnen
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # Buttons
        self.predict_btn = tk.Button(self.root, text="Erkennen", command=self.predict, width=15, height=2, bg='green', fg='white')
        self.predict_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.clear_btn = tk.Button(self.root, text="Löschen", command=self.clear_canvas, width=15, height=2, bg='red', fg='white')
        self.clear_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Result Label
        self.result_label = tk.Label(self.root, text="Zeichne einen Buchstaben (A-Z)!", font=("Arial", 18))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Confidence Frame
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Instructions
        self.info_label = tk.Label(self.root, text="Tipp: Zeichne große, klare Buchstaben in der Mitte", font=("Arial", 10), fg='gray')
        self.info_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Maus-Events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.old_x = None
        self.old_y = None
        
        # Image zum Speichern
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Model laden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LetterConvNet(num_classes=26).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform für Input (EMNIST spezifisch)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))
        ])
        
    def paint(self, event):
        """Malen auf Canvas"""
        if self.old_x and self.old_y:
            # Auf Canvas zeichnen
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=10, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            # Auf PIL Image zeichnen
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill='black', width=10)
        
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
        self.result_label.config(text="Zeichne einen Buchstaben (A-Z)!")
        
        # Clear confidence labels
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        
    def predict(self):
        # 1. Invertiere das Canvas-Bild (weiß auf schwarz → schwarz auf weiß)
        img = Image.eval(self.image, lambda x: 255 - x)

        # 2. Bounding Box finden
        bbox = img.getbbox()
        if not bbox:
            self.result_label.config(text="Kein Buchstabe erkannt!")
            return

        # 3. Mit Padding zuschneiden (ähnlich wie EMNIST)
        padding = 10
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(self.canvas_size, right + padding)
        lower = min(self.canvas_size, lower + padding)
        img = img.crop((left, upper, right, lower))

        # 4. In quadratisches Format bringen
        img_width, img_height = img.size
        max_dim = max(img_width, img_height)

        # Neues quadratisches Bild erstellen (weißer Hintergrund)
        square_img = Image.new('L', (max_dim, max_dim), 0)  # weiß
        offset = ((max_dim - img_width) // 2, (max_dim - img_height) // 2)
        square_img.paste(img, offset)
        img = square_img

        # 5. Skalieren auf 28x28 (wie EMNIST)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 6. Transform anwenden
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 7. Vorhersage
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 8. Anzeige
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        predicted_letter = alphabet[predicted.item()]
        self.result_label.config(
            text=f"Erkannter Buchstabe: {predicted_letter} ({confidence.item()*100:.1f}%)"
        )

        # Top 5 anzeigen
        top5_prob, top5_idx = torch.topk(probabilities[0], 5)
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
            letter = chr(idx.item() + ord('A'))
            color = 'green' if i == 0 else 'black'
            label = tk.Label(self.confidence_frame, text=f"{letter}: {prob.item()*100:.1f}%", fg=color)
            label.grid(row=0, column=i, padx=5)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    import os
    if not os.path.exists('emnist_letters_model.pth'):
        print("Fehler: emnist_letters_model.pth nicht gefunden!")
        print("Bitte erst das EMNIST Letters Modell trainieren.")
        print("Führe 'python emnist_letters_training.py' aus.")
    else:
        app = LetterDrawingApp()
        app.run()