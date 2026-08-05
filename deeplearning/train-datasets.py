#Training datasets using pytorch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#define transformation with data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root = "./data", train = True, download = True, transforms = transform_train)
test_dataset = datasets.CIFAR10(root = "./data", train = True, download = True, transform=transform_train)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)

print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

class EnhancedCNN (nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0,5)
        
        #calculate the size of the output
        self._calculate_conv_output()

        self.fc1 = nn.Linear(self.conv_output_size, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)




