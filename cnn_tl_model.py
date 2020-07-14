################################################################################################################################
# ebharucha: 14/7/2020
################################################################################################################################

# Import dependencies
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import prep_data
import matplotlib.pyplot as plt

# Function to plot Accuracy & Loss
def plot_acc_loss(train_test, accuracy, loss_):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(accuracy)
    plt.title (f'{train_test} Accuracy')
    plt.subplot(1,2,2)
    plt.plot(loss_)
    plt.title (f'{train_test} Loss')
    plt.show()

# Function to train model
def train_model(model, num_epochs, dataloader, device, criterion, optimizer):
    n_correct = 0
    train_accuracy = []
    train_loss = []
    total_samples = 0
    model.train()
    for epoch in range(num_epochs):
        for i, (X_train, y_train) in tqdm(enumerate(dataloader)):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # Forward pass
            yhat_train = model(X_train)
            loss = criterion(yhat_train, y_train)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute accuracy
            _, predicted = torch.max(yhat_train.data, 1)
            n_correct += (predicted == y_train).sum().item()
            total_samples+=len(y_train)
        
        # Loss & accuracy per epoch
        train_loss.append(loss.item())
        train_accuracy.append(n_correct/total_samples)        
        print(f'Epoch {epoch+1}/{num_epochs} => Training Loss: {loss.item():.4f}')
    
    plot_acc_loss(train_test='Train', accuracy=train_accuracy, loss_=train_loss)

    return (model)

# Function to test model
def test_model(model, dataloader, device, criterion):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        test_accuracy = []
        test_loss = []
        total_samples = 0
        for i, (X_test, y_test) in enumerate(dataloader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            yhat_test = model(X_test)
            loss = criterion(yhat_test, y_test)
            
            # Compute accuracy
            _, predicted = torch.max(yhat_test.data, 1)
            n_correct += (predicted == y_test).sum().item()
            total_samples+=len(y_test)
            
            # Append loss & accuracy
            test_loss.append(loss.item())
            test_accuracy.append(n_correct / total_samples)
            
        # Print Test Accuracy & Loss
        print (f'Test Accuracy: {test_accuracy[-1]:.4f}')
        print (f'Test Loss: {test_loss[-1]:.4f}')

    plot_acc_loss(train_test='Test', accuracy=test_accuracy, loss_=test_loss)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def choose_model(option):
    classes = prep_data.traintest.classes
    if (option == 'resnet18'):
        print ('Transfer Learning using Resnet18')
        # Load pre-trained model
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))  # Add layer with 5 outputs (our classes)
    elif (option == 'vgg16'):
        # https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
        print ('Transfer Learning using VGG16')
        model = torchvision.models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(classes))]) # Add layer with 5 outputs (our classes)
        model.classifier = nn.Sequential(*features) # Replace the model classifier
    return (model)

# Choose model
# model = choose_model('resnet18')
model = choose_model('vgg16')
model = model.to(device)

# Hyper-parameters
num_epochs = 10
batch_size = 5
learning_rate = 0.001

# Define Loss & Optimzier
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train model        
model = train_model(model, num_epochs, prep_data.dataloader_train, device, criterion, optimizer)
# Test model
test_model(model, prep_data.dataloader_test, device, criterion)

# MODELPATH = './models/cnn_resnet18_1.pt'
MODELPATH = './models/cnn_vgg16_1.pt'
torch.save(model, MODELPATH)

