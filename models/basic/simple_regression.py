import torch
import torch.nn as nn
import random
import math
import numpy as np
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden1 = torch.tanh(self.hidden_layer1(x))
        hidden2 = torch.tanh(self.hidden_layer2(hidden1))
        output = self.output_layer(hidden2)
        return output
    

if __name__ == "__main__":
    # * Sin(x)
    x_train = [random.uniform(-2*math.pi, 2*math.pi) for _ in range(1000)]
    y_train = [math.sin(x) for x in x_train]

    # * Cos(x)
    # x_train = [random.uniform(-2*math.pi, 2*math.pi) for _ in range(1000)]
    # y_train = [math.cos(x) for x in x_train]

    x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = SimpleMLP(input_size=1, hidden_size=128, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(700):
        model.train()
        optimizer.zero_grad()
        
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    with torch.no_grad():
        test_x = torch.linspace(-2*math.pi, 2*math.pi, 100).unsqueeze(1)
        predictions = model(test_x)
        true_values = torch.sin(test_x)
        
        plt.plot(test_x.squeeze(), predictions.squeeze(), 'r-', label='Predicted')
        plt.plot(test_x.squeeze(), true_values.squeeze(), 'b-', label='True sin(x)')
        plt.legend()
        plt.show()