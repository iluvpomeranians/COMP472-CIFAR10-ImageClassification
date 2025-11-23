#TODO: MLP algorithm


#inputs-> transfer function: sum(weights*x1+b)->activation function: 1/(1+e^-x)*1-(1/(1+e^-x)) 
#backpropagate training

# iterations until error rate is minimized - all data must be evaluated at least for an epoch

#linear(50,512)-ReLu
#linear(512,512)-batchNorm(512)-ReLU
#linear(512,10)

# input layer-> 2 hidden layers-> output layer 

#cross entropy loss with torch.nn.CrossEntropyLoss momentum=0.9


# Sources
# https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d
# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
# https://www.kaggle.com/code/pinocookie/pytorch-simple-mlp
# https://www.youtube.com/watch?v=tJ3-KYMMOOs

#-----------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score




def load_50npz():
    data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    print (f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
    print (f"Label ranges: train[{y_train.min()}..{y_train.max()}], test[{y_test.min()}..{y_test.max()}]")
    return X_train, y_train, X_test, y_test



class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.functions=nn.Sequential(

            nn.Linear(50,512),nn.ReLU(),


            nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),


            nn.Linear(512,10),
        )

    def forward(self,x):
        self.functions(x)


        return self.functions(x)
    
    

def mlp_training(model, device="cuda", epoch_num=20):                       #https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
    print(f"MLP Training with {device} in progress. ")
    model.to(device)

    X_train,y_train, X_test, y_test = load_50npz
    X_train=torch.tensor(X_train, dtype=torch.float32)
    y_train=torch.tensor(y_train, dtype=torch.long)
    X_test=torch.tensor(X_test, dtype=torch.float32)
    y_test=torch.tensor(y_test, dtype=torch.long)

    train_load=DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_load=DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    criterion= nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    print(f"Epoch number {epoch_num} ")

    for epoch in range(epoch_num):
        model.train()



    







if __name__=="__main__":
    mlp_training()