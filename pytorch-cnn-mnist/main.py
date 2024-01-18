import torch
from torch.optim import Adam
from torch import nn, save, load
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import ImageClassifier
from PIL import Image

# Training code

# train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
# dataset = DataLoader(train_data, batch_size=32, shuffle=True)

classifier = ImageClassifier().to('cpu')
# optimizer = Adam(classifier.parameters(), lr=1e-3)
# loss_function = nn.CrossEntropyLoss()

"""
for epoch in range(2):
    for batch in dataset:
        inputs, targets = batch
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        outputs = classifier(inputs)
        loss = loss_function(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

with open('model.pt', 'wb') as f:
    save(classifier.state_dict(), f)
"""

# Inference code
with open('model.pt', 'rb') as f:
    classifier.load_state_dict(load(f))

img = Image.open('test.jpeg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

print(torch.argmax(classifier(img_tensor)))