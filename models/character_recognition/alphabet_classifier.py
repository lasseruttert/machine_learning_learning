import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LetterConvNet(nn.Module):
    def __init__(self, num_classes=26):
        super(LetterConvNet, self).__init__()
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
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
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

if __name__ == "__main__":
    # Parameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 20
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Custom transform class für Windows-Kompatibilität
    class TransposeTransform:
        def __call__(self, x):
            return x.transpose(1, 2)
    
    # EMNIST benötigt eine spezielle Transform wegen der Rotation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Mehr Rotation für Buchstaben
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Mehr Verschiebung
        transforms.ToTensor(),
        TransposeTransform(),  # EMNIST ist transponiert
        transforms.Normalize((0.1751,), (0.3332,))  # EMNIST spezifische Werte
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeTransform(),  # EMNIST ist transponiert
        transforms.Normalize((0.1751,), (0.3332,))
    ])
    
    # Datasets - EMNIST Letters (nur A-Z, keine Kleinbuchstaben)
    try:
        train_dataset = datasets.EMNIST(
            root='./data',
            split='letters',  # 'letters' split hat 26 Klassen (A-Z)
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = datasets.EMNIST(
            root='./data',
            split='letters',
            train=False,
            download=True,
            transform=test_transform
        )
        
        # Klassen anpassen (EMNIST letters beginnt bei 1, nicht 0)
        train_dataset.targets = train_dataset.targets - 1
        test_dataset.targets = test_dataset.targets - 1
        
    except Exception as e:
        print(f"Fehler beim Laden von EMNIST: {e}")
        print("Installiere torchvision >= 0.7.0 oder verwende einen anderen Datensatz")
        exit(1)
    
    print(f"Anzahl Trainingsbilder: {len(train_dataset)}")
    print(f"Anzahl Testbilder: {len(test_dataset)}")
    print(f"Anzahl Klassen: {len(train_dataset.classes)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 0 für Windows!
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # 0 für Windows!
    )
    
    # Model, Loss, Optimizer
    model = LetterConvNet(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    # Training
    print("\nStarting training...")
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Test evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        class_correct = list(0. for i in range(26))
        class_total = list(0. for i in range(26))
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                c = (predicted == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        test_accuracy = 100 * test_correct / test_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
        
        # Speichere bestes Modell
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'emnist_letters_model.pth')
            print(f"Neues bestes Modell gespeichert! Accuracy: {test_accuracy:.2f}%")
        
        scheduler.step()
    
    # Zeige Per-Class Accuracy
    print("\nPer-Class Test Accuracy:")
    for i in range(26):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{chr(i+65)}: {accuracy:.1f}%', end='  ')
            if (i+1) % 6 == 0:
                print()  # Neue Zeile alle 6 Buchstaben
    
    print(f"\n\nTraining complete! Best test accuracy: {best_accuracy:.2f}%")
    print("Model saved as 'emnist_letters_model.pth'")