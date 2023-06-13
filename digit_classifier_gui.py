import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class DigitClassifierGUI(tk.Tk):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.canvas_size = 300
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Initialize the UI elements
        self.init_ui()

    def init_ui(self):
        self.canvas = self.create_canvas()
        self.classify_button = self.create_classify_button()
        self.clear_button = self.create_clear_button()
        self.prediction_text = tk.StringVar()
        self.prediction_label = tk.Label(self, textvariable=self.prediction_text)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky='W')
        self.classify_button.grid(row=0, column=1, pady=2, padx=2)
        self.clear_button.grid(row=0, column=2, pady=2, padx=2)
        self.prediction_label.grid(row=1, column=0, pady=2, padx=2)

    def create_canvas(self):
        canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white', cursor='cross')
        canvas.bind("<Button-1>", self.save_last_point)
        canvas.bind("<B1-Motion>", self.draw_lines)
        return canvas

    def create_classify_button(self):
        return tk.Button(self, text="Classify", command=self.classify_handwriting)

    def create_clear_button(self):
        return tk.Button(self, text="Clear", command=self.clear_all)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def classify_handwriting(self):
        img = self.image.resize((28, 28))
        img = 255 - np.array(img)  # invert colors
        img = img / 255.0  # scale values between 0 and 1
        img = img.reshape(1,28,28,1)
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            output = self.model(img)
        ps = torch.exp(output)
        probab = list(ps.numpy()[0])
        _, predicted = torch.max(output, 1)
        pred = predicted.item()
        confidence = probab[pred]
        self.prediction_text.set('Prediction: {}'.format(pred, confidence * 100))

    def save_last_point(self, event):
        self.lastx, self.lasty = event.x, event.y

    def draw_lines(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), fill='black', width=8, capstyle='round', smooth=True)
        self.draw.line((self.lastx, self.lasty, event.x, event.y), fill='black', width=8)
        self.lastx, self.lasty = event.x, event.y

# initialize and run the app
model = Network()  # ensure that the model is defined and loaded with weights
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()
app = DigitClassifierGUI(model)
app.mainloop()
