import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_dataset = datasets.ImageFolder("Q5_train/validation_dataset", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = models.resnet50(pretrained=True)
# Replace the output layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 1),
    torch.nn.Sigmoid()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU") if torch.cuda.is_available() else print("Using CPU")

def compare_train_model(model_path_1, model_path_2):
    # model 1 : ResNet50 without RandomErasing
    print(f"testing model 1 ... (ResNet50 without RandomErasing)  path: {model_path_2}")
    model.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels.float().view(-1, 1)).sum().item()
    test_accuracy1 = 100 * correct_test / total_test # (%)

    # model 2 : ResNet50 with RandomErasing
    print(f"testing model 2 ... (ResNet50 with RandomErasing)  path: {model_path_2}")
    model.load_state_dict(torch.load(model_path_2, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels.float().view(-1, 1)).sum().item()
    test_accuracy2 = 100 * correct_test / total_test # (%)

    # 設置柱狀圖的數據
    accuracies = [test_accuracy1, test_accuracy2]
    models = ['Without Random Erasing', 'With Random Erasing']
    
    plt.bar(models, accuracies, color=['blue', 'blue'], width=0.5)
    plt.title('Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.5, str(int(v)), ha='center', va='bottom')
    plt.savefig('Q5_train/Accuracy_Comparison.png')
    plt.show()
    print("Finish Comparison!")


if __name__ == '__main__':
    model_path_1 = "model/ResNet50_wo_batch=32.pth"
    model_path_2 = "model/ResNet50_w_batch=5.pth"
    compare_train_model(model_path_1, model_path_2)