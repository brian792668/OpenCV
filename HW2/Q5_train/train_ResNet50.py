import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torchsummary
import matplotlib.pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_ResNet50(batch, train_epoch, random_erasing):
    if random_erasing == True:
        print("Train model with random_erasing")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
        ])
    else:
        print("Train model with no random_erasing")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    train_dataset = datasets.ImageFolder("Q5_train/training_dataset", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataset = datasets.ImageFolder("Q5_train/validation_dataset", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Training model (ResNet50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Train on GPU") if torch.cuda.is_available() else print("Train on CPU")
    model = models.resnet50(pretrained=True)
    # Replace the output layer
    model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )
    model.to(device)

    # Show model structure
    torchsummary.summary(model, (3, 224, 224))  # assuming input size is (3, 224, 224)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)

    # Initialize parameters
    t0 = time.time()
    train_losses = []
    train_accuracy = []
    test_losses = []
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

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_temp += loss.item()

            predicted = (outputs >= 0.5).float()
            correct_train += (predicted == labels.float().view(-1, 1)).sum().item()
            total_train += labels.size(0)

            if i%100 == 99:
                print(f"[ epoch: {epoch + 1}/{train_epoch} - {i*100*batch/16200:.0f} %  loss: {loss_temp/100:.2f} ]  {(time.time()-t0)/3600.0:.2f} hr")
                loss_temp = 0.0

        # Calculate loss & accuracy on [train data] in each epoch
        train_accuracy.append(100 * correct_train / total_train)
        train_losses.append(running_loss / len(train_loader))

        # Validate model using [val_dataset] in each epoch
        model.eval()
        correct_test = 0
        total_test = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predicted = (outputs >= 0.5).float()
                loss = criterion(outputs, labels.float().view(-1, 1))
                running_loss += loss.item()

                total_test += labels.size(0)
                correct_test += (predicted == labels.float().view(-1, 1)).sum().item()

        # Calculate loss & accuracy (validation)
        test_accuracy.append(100 * correct_test / total_test)
        test_losses.append(running_loss / len(val_loader))
        epoch_plot.append(epoch+1)

        # Save the best model when best accuracy occurs
        if ( test_accuracy[-1] >= best_accuracy):
            if random_erasing == True:
                PATH = f'model/ResNet50_w_batch={batch}.pth'
            else :
                PATH = f'model/ResNet50_wo_batch={batch}.pth'
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
        if random_erasing == True:
            plt.savefig(f'Q5_train/ResNet50_w_batch={batch}.png')
        else:
            plt.savefig(f'Q5_train/ResNet50_wo_batch={batch}.png')
        plt.close()

if __name__ == '__main__':
    train_ResNet50(batch=100, train_epoch=30, random_erasing = True)