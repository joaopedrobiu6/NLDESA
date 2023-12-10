import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

HLS_1 = 10
HLS_2 = 30
HLS_3 = 10
MAX_EPOCHS = 10
# Define the network
class ODEStabilityNN(nn.Module):
    def __init__(self, n_classes, n_features, hidden_size1, hidden_size2, hidden_size3, dropout_prob=0):
        super().__init__()
        
        # ChatGPT Architecture LMAO 
        self.layer1 = nn.Linear(n_features, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.Tanh()
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(hidden_size2, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x
    
def train(model, X, y, optimizer, loss_fn, epochs=100):
    """
        Defining the training loop. 
    """
    model.train()  # <-- here
    loss_history = []
    for i in range(epochs):
        print(f"Training Epoch {i+1}...")
        optimizer.zero_grad() # sets the gradients "to zero".

        y_ = model(X)
        loss = loss_fn(y_, y)
        
        loss_history.append(loss.item())

        loss.backward() # computes the gradients.
        optimizer.step() # updates weights using the gradients.

    return loss_history

def evaluate(model, X):
    """
        Evaluating the model. 
    """
    model.eval()  # <-- here
    with torch.no_grad(): 
        y_ = model(X)    
    return y_

def accuracy(y, y_hat):
    """
        Accuracy of the model. 
    """
    print(y.shape)
    return np.mean((y == y_hat).numpy())

def train_model():
    #Load data 
    data = np.load('../data/training_set.npz')

    features = data['features']
    labels = data['labels']

    X = torch.from_numpy(features).float()
    Y = torch.tensor(labels.reshape((X.shape[0], 1)),dtype=torch.float)

    # Separate into training and test data
    train_size = int(1 * X.shape[0])
    test_size = X.shape[0] - train_size

    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of Y_train: {Y_train.shape}")
    print(f"Shape of Y_test: {Y_test.shape}")
    
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    print(f"Number of Features: {n_features}")
    print(f"Number of Classes: {n_classes}")

    model = ODEStabilityNN(n_classes,n_features,hidden_size1=HLS_1,hidden_size2=HLS_2,hidden_size3=HLS_3)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    loss_history = train(model, X_train, Y_train, optimizer, loss_fn, epochs=MAX_EPOCHS)
    
    # Save model parameters
    torch.save(model.state_dict(), '../data/ode_stability_model.pth')

    # Plot Loss
    plt.plot(loss_history)
    plt.title('Loss per epoch')
    plt.show()
        
def plot_neural_model():

    # Generate features    
    x = np.linspace(-25,25,1000)
    y = np.linspace(-25,25,1000)
    X,Y = np.meshgrid(x,y)
    features = np.column_stack((X.flatten(), Y.flatten()))    
    features = torch.from_numpy(features).float()
    
    n_features = features.shape[1]
    n_classes = 1
    
    # Load Model
    model = ODEStabilityNN(n_classes,n_features,hidden_size1=HLS_1,hidden_size2=HLS_2,hidden_size3=HLS_3)
    model.load_state_dict(torch.load('../data/ode_stability_model.pth'))
    #model.eval()
    
    labels = evaluate(model,features) 
    
    #Plot Model
    Z = labels.reshape(len(y),len(x))

    plt.pcolormesh(X,Y,Z)

    plt.show()


if __name__ == "__main__":  
    #train_model()
    plot_neural_model()