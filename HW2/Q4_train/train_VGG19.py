import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchsummary
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_VGG(batch, train_epoch):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset = MNIST(
        root="./Q4_train", 
        train=True,
        download=True, 
        transform=transform)
    trainloader = DataLoader(
        trainset, 
        batch_size=batch, 
        shuffle=True, 
        num_workers=2)
    testset = MNIST(
        root="./Q4_train", 
        train=False,
        download=True,
        transform=transform)
    testloader = DataLoader(
        testset,
        batch_size=batch,                
        shuffle=False,
        num_workers=2)
    
    # Training model (VGG19)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Train on GPU") if torch.cuda.is_available() else print("Train on CPU")
    model = models.vgg19_bn(num_classes=10)
    model.to(device)

    # Show model structure
    torchsummary.summary(model, (3, 32, 32))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    # optimizer = optim.SGD(model.parameters(), lr=0.000005, momentum=0.9)

    # Initialize parameters
    t0 = time.time()
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    epoch_plot = []
    best_accuracy = 0
    
    for epoch in range(train_epoch):  # loop over the dataset multiple times
        # Train model using [train_dataset]
        model.train()
        running_loss = 0.0
        loss_temp = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_temp += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            if i%50 == 49:
                print(f"[ epoch: {epoch + 1}/{train_epoch} - {i*100*batch/60000:.0f} %  loss: {loss_temp/100:.2f} ]  {(time.time()-t0)/3600.0:.2f} hr")
                loss_temp = 0.0
        
        # calculate loss & accuracy on [train data] in each epoch
        train_accuracy.append(100 * correct_train / total_train)
        train_losses.append(running_loss / len(trainloader))

        # Validate model using [test data] in each epoch
        model.eval()
        correct_test = 0
        total_test = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # calculate loss & accuracy on [test data]
        test_accuracy.append(100 * correct_test / total_test)
        test_losses.append(running_loss / len(testloader))
        epoch_plot.append(epoch+1)

        # Save the best model when best accuracy occurs
        if ( test_accuracy[-1] >= best_accuracy):
            PATH = f'model/VGG19_MNIST_batch={batch}.pth'
            torch.save(model.state_dict(), PATH)

        # Plot training figure and save
        plt.figure(figsize=(10, 10))
        plt.subplot(2,1,1)
        plt.plot(epoch_plot, train_losses, label='Train Loss')
        plt.plot(epoch_plot, test_losses, label='Test Loss')
        plt.xlim(-2, train_epoch+2)
        plt.ylim(0, None)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')

        plt.subplot(2,1,2)
        plt.plot(epoch_plot, train_accuracy, label='train_acc')
        plt.plot(epoch_plot, test_accuracy, label='val_acc')
        plt.xlim(-2, train_epoch+2)
        plt.ylim(0, 100)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.title('Accuracy vs. Epoch')

        plt.savefig(f'Q4_train/VGG19_MNIST_batch={batch}.png')
        plt.close()

if __name__ == '__main__':
    train_VGG(batch = 250, train_epoch = 30)