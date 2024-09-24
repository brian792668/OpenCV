import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchsummary
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training on", device)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = 8

        trainset = CIFAR10(
                root='./data', 
                train=True,
                download=True, 
                transform=transform)
        trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2)
        testset = CIFAR10(
                root='./data', 
                train=False,
                download=True, 
                transform=transform)
        testloader = DataLoader(
                testset,
                batch_size=batch_size,                
                shuffle=False,
                num_workers=2)

        model = models.vgg19_bn(num_classes=10)
        model.to(device)
        # torchsummary.summary(model, (3, 32, 32))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0015, momentum=0.9)

        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
        epoch_plot = []
        t0 = time.time()
        train_epoch = 50

        for epoch in range(train_epoch):  # loop over the dataset multiple times
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
                loss_temp +=loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if i%100 == 99:
                        print(f"[ epoch: {epoch + 1}/{train_epoch} - {i*100*batch_size/50000:.0f} %  loss: {loss_temp/100:.2f} ]  {(time.time()-t0)/3600.0:.2f} hr")
                        loss_temp = 0.0
            
            # calculate loss & accuracy on train data
            train_accuracy.append(100 * correct_train / total_train)
            train_losses.append(running_loss / len(trainloader))

            # calculate loss & accuracy on test data
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

            test_accuracy.append(100 * correct_test / total_test)
            test_losses.append(running_loss / len(testloader))
            epoch_plot.append(epoch+1)

        print('Finished Training')
        PATH = './VGG19_cifar_net_batch6.pth'
        torch.save(model.state_dict(), PATH)

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
        plt.show()


if __name__ == '__main__':
    main()