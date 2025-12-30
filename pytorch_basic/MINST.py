import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

def load_data(Batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root = './data',train = True, download = True,
        transform = transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root = './data', train = False, download = True,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size = Batch_size,shuffle = True
    )
    test_loader = DataLoader(
        test_dataset, batch_size = Batch_size, shuffle = False
    )

    return train_loader, test_loader

class models(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 128, num_classes = 10):
        super(models,self).__init__()
        
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device,num_epochs = 5):
    model.train()
    train_loss = [] 
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}')

    return train_loss

def eval_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = [] 
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels,all_preds)
    print(f'test accuracy: {accuracy:.4f}')
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
    print(device)

    batch_size = 64
    lr = 0.001
    num_epochs = 5
    train_loader , test_loader = load_data(batch_size)
    model = models().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    print ('##### the train is started #####')
    train_losses = train_model(model,train_loader,criterion,optimizer,device,num_epochs)

    print('##### the model evluation #####')
    test_accuracy = eval_model(model, test_loader, device)

    plt.plot(range(1,num_epochs + 1),train_losses)
    plt.title('the varaity of train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    torch.save(model.state_dict(),'modelofMNIST.pth')
    print("the model has been saved as modelofMNIST")

if __name__ == '__main__':
    main()