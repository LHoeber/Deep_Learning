#!/usr/bin/env python
# coding: utf-8

# In[1]:


#preliminary setup
# %pip install -r requirements.txt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler


#Fetching the data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class NeuralNet_default(nn.Module):
    def __init__(self):
        super(NeuralNet_default, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)

        return x


class NeuralNet_deep(nn.Module):
    def __init__(self):
        super(NeuralNet_deep, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x


class NeuralNet_wide(nn.Module):
    def __init__(self):
        super(NeuralNet_wide, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)

        return x


class NeuralNet_deeper_wide(nn.Module):
    def __init__(self):
        super(NeuralNet_deeper_wide, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return x


class NeuralNet_deep_wider(nn.Module):
    def __init__(self):
        super(NeuralNet_deep_wider, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x


class NeuralNet_deeper_wide_classification(nn.Module):
    def __init__(self):
        super(NeuralNet_deeper_wide_classification, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs

# set seeds for reproducibility
# random behaviours taken from https://www.geeksforgeeks.org/reproducibility-in-pytorch/
random_seed = 302
# torch.manual_seed(302); np.random.seed(302)

np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=random_seed)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=random_seed, )


# In[2]:


ONLY_TRAINING_SET = False
#a.) Investigating the dataset

all_sets = [X_train,y_train,X_validation,y_validation,X_test,y_test]
all_sets_names = ["Training set","Training targtets","Validation set","Validation targets","Test set","Test targets"]
features = ["MedInc","HouseAge","AveRooms","AveBdrms","Population","AveOccup","Latidue","Longitude"]
if ONLY_TRAINING_SET:
    all_sets = [X_train, y_train]
    all_sets_names = ["Training set","Training targtets"]


#Checking the data, to see what would be good to normalize
for set, name in zip(all_sets[0::2],all_sets_names[0::2]):
    
    print('\033[4m'+'\033[1m'+f"{name}"+'\033[0m'+'\033[0m')
    print(f"size: {str(set.shape) : >10}")
    stats = {"mean":[],
             "std":[],
             "min":[],
             "max":[]}
    fig,axs = plt.subplots(2,len(features)//2,figsize=(10,4),layout = "tight")
    for col,feature in enumerate(features):
        feature_data = set[:,col]
        stats["mean"].append(np.mean(feature_data))
        stats["std"].append(np.std(feature_data))
        stats["min"].append(np.min(feature_data))
        stats["max"].append(np.max(feature_data))
        axs[int(np.floor(col/4))][(col%4)].hist(feature_data,bins=20)
        axs[int(np.floor(col/4))][(col%4)].set_title(feature)
        

    print('\033[1m'+"          "+f"{'   '.join(features)}"+'\033[0m')
    for stat,vals in stats.items():
        x = [str(round(val,2)) for val in vals]
        print('{:>4}{:>12s}{:>11s}{:>11s}{:>11s}{:>12s}{:>12s}{:>11s}{:>11s}'.format(stat,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]))
    
    plt.suptitle("Original feature distributions, training data")
    plt.show()
    # fig.savefig(f"./plots/feature distributions_train.png")

#Normalizing data

#fitting the scalers first on the training set and then applying same scalings to the validation and test set
normalized_sets = []
standard_dist_scaler_1 = StandardScaler()
standard_dist_scaler_1.fit(X_train)
X_train_standard = standard_dist_scaler_1.transform(X_train)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train_standard)
X_train_standart_minmax = min_max_scaler.transform(X_train_standard)

#do another scaling, so it's centered around 0 again
standard_dist_scaler_2 = StandardScaler()
standard_dist_scaler_2.fit(X_train_standart_minmax)

#TODO: Decide normalization on individual level
# now only applied standardization everywhere
names = ["training","validation", "test"]

for i,set in enumerate(all_sets[0::2]):
    norm_set = standard_dist_scaler_1.transform(set)
    #norm_set = min_max_scaler.transform(norm_set)
    #norm_set = standard_dist_scaler_2.transform(norm_set)
    normalized_sets.append(norm_set)

    # new_set = np.zeros_like(set)
    # for col in range(len(features)):

    #     norm_feature = np.reshape(set[:,col],(1,len(set[:,col])))
        
    #     #Standard gauss distribution
    #     norm_feature = (norm_feature-np.mean(norm_feature))/np.std(norm_feature)
    #     all_sets[i*2][:,col] = norm_feature
        
    #     #min-max scaling into -1 +1 range
    #     norm_feature = (norm_feature-np.min(np.abs(norm_feature)))/(np.max(np.abs(norm_feature))-np.min(np.abs(norm_feature)))
    #     new_set[:,col] = norm_feature

    # normalized_sets.append(new_set)

for i,(set,name) in enumerate(zip(normalized_sets,all_sets_names[0::2])):
    print('\033[4m'+'\033[1m'+f"{name}"+'\033[0m'+'\033[0m')
    print(f"size: {str(set.shape) : >10}")
    stats = {"mean":[],
                "std":[],
                "min":[],
                "max":[]}
    fig,axs = plt.subplots(2,len(features)//2,figsize=(10,4),layout = "tight")
    for col,feature in enumerate(features):
        feature_data = set[:,col]
        stats["mean"].append(np.mean(feature_data))
        stats["std"].append(np.std(feature_data))
        stats["min"].append(np.min(feature_data))
        stats["max"].append(np.max(feature_data))
        axs[int(np.floor(col/4))][(col%4)].hist(feature_data,bins=20)
        axs[int(np.floor(col/4))][(col%4)].set_title(feature)
        

    print('\033[1m'+"          "+f"{'   '.join(features)}"+'\033[0m')
    for stat,vals in stats.items():
        x = [str(round(val,2)) for val in vals]
        print('{:>4}{:>12s}{:>11s}{:>11s}{:>11s}{:>12s}{:>12s}{:>11s}{:>11s}'.format(stat,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]))

    plt.suptitle(f"Normalized feature distributions, {names[i]} data")
    plt.show()
    # fig.savefig(f"./plots/normalized feature distributions_{names[i]}.png")


# In[3]:


len(all_sets_names)


# ## Networks

# In[4]:


# from models import NeuralNet_deep, NeuralNet_wide, NeuralNet_default, NeuralNet_deep_wider, NeuralNet_deeper_wide

def train_model(model, train_loader, val_loader=None, device='cpu', optimizer=None, num_epochs=30, scheduler=None):
    torch.manual_seed(302); np.random.seed(302)
    loss_fn = nn.MSELoss()
    #saves losses on every batch
    train_losses_all = []
    val_losses_all = []
    model.to(device)
    for epoch in range(num_epochs):
        train_losses_epoch = []
        print('-'*20, f'Epoch {epoch}', '-'*20)
        # Train one epoch
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if optimizer is not None:
                optimizer.zero_grad()
            predict = model(data)
            loss = loss_fn(predict, target)
            loss.backward()
            if optimizer is not None:
                optimizer.step()

            train_losses_epoch.append(loss.item())
        if scheduler is not None:
            scheduler.step() # update learning rate based on the schedule
        train_loss_epoch_mean =np.mean(train_losses_epoch)
        train_losses_all.append(train_loss_epoch_mean)
        print(f'Average Training Loss for epoch(over all batches) {np.mean(train_losses_epoch)}')
        

        # correct = 0
        if val_loader is not None:
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    predict = model(data)
                    val_loss += F.mse_loss(predict, target, reduction='sum').item()  # sum up batch loss
                    # correct += (predict == target).sum().item()

            val_loss /= len(val_loader.dataset)
            # avg_correct = correct / len(val_loader.dataset)
            val_losses_all.append(val_loss)
            # val_accuracies.append(avg_correct)

            print(f'Validation loss for epoch: {val_loss:.4f})\n')
    
    return train_losses_all, val_losses_all


# In[5]:


#evaluation of the different models
from torch.utils.data import TensorDataset, DataLoader

X_train_norm = normalized_sets[0]
X_val_norm = normalized_sets[1]
X_test_norm = normalized_sets[2]

random_seed = 302
torch.manual_seed(302); np.random.seed(302)

X_train_torch = torch.tensor(X_train_norm,dtype=torch.float32)
y_train_torch = torch.tensor(y_train.reshape((len(y_train),1)),dtype=torch.float32)
X_validation_torch = torch.tensor(X_val_norm,dtype=torch.float32)
y_validation_torch = torch.tensor(y_validation.reshape((len(y_validation),1)),dtype=torch.float32)
X_test_torch = torch.tensor(X_test_norm,dtype=torch.float32)
y_test_torch = torch.tensor(y_test.reshape((len(y_test),1)),dtype=torch.float32)

train_dataset = TensorDataset(X_train_torch,y_train_torch)
validation_dataset = TensorDataset(X_validation_torch,y_validation_torch)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_seed))
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_seed))

device = 'cuda' if torch.cuda.is_available() else 'cpu' # if you have a gpu, you can move the model onto it like this
all_models = NeuralNet_deep(), NeuralNet_wide(), NeuralNet_default(), NeuralNet_deep_wider(), NeuralNet_deeper_wide()

best_loss = 100
for model in all_models:
    print(model._get_name())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = train_model(model,train_loader,val_loader,device, optimizer=optimizer, num_epochs=30)

    if np.min(val_loss) < best_loss:
        best_model = model
        best_epoch = np.argmin(val_loss)
        best_loss = np.min(val_loss)

    fig,axs = plt.subplots(1,2,layout="tight")
    axs[0].plot(train_loss)
    axs[0].set_title("training loss")
    # axs[0].set_ylim([max(train_loss),max(train_loss)])
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("MSE loss")
    
    axs[1].plot(val_loss)
    axs[1].set_title("validation loss")
    # axs[1].set_ylim([max(val_loss),max(val_loss)])
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("MSE loss")
    plt.suptitle(f"{model._get_name()}")
    plt.show()

print(f"best model: {best_model._get_name()}, best epoch: {best_epoch}, best validation loss: {best_loss}")


# ## c) Optimizers and Scheduling

# In[6]:


# best model: NeuralNet_deeper_wide, best epoch: 28, best validation loss: 0.2711376182792723
# best model: NeuralNet_deeper_wide, best epoch: 28, best validation loss: 0.2775191000001639 aktuellstes
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
best_model = NeuralNet_deeper_wide()
torch.manual_seed(302); np.random.seed(302) # for safety reasons
lr_rates = [0.001, 0.005, 0.01]

train_val_loss_001 = []
train_val_loss_005 = []
train_val_loss_01 = []

best_combi = {
    'optimizer': None,
    'scheduler': None,
    'lr': None,
    'val_loss': 100,
    'train_loss': None,
    'epoch': None}


schedulers = {
    'StepLR': lambda optimizer: StepLR(optimizer, step_size=15, gamma=0.5),
    'MultiStepLR': lambda optimizer: MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.5),
    'CosineAnnealingLR': lambda optimizer: CosineAnnealingLR(optimizer, T_max=30),
    'CosineAnnealingWarmRestarts': lambda optimizer: CosineAnnealingWarmRestarts(optimizer, T_0=15),
}

for lr in lr_rates:
    optimizers = {
        'SGD': lambda params: torch.optim.SGD(params, lr=lr, momentum=0),
        'SGD with Momentum': lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9),
        'Nesterov SGD with Momentum': lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
        'RMSProp': lambda params: torch.optim.RMSprop(params, lr=lr),
        'Adam': lambda params: torch.optim.Adam(params, lr=lr)
    }


    for optimizer_name, optimizer_init_fn in optimizers.items():
        for scheduler_name, scheduler_init_fn in schedulers.items():
            model = best_model
            model.to(device)
            # reinstantiating should reset optimizer parameters
            optimizer = optimizer_init_fn(model.parameters())
            scheduler = scheduler_init_fn(optimizer)

            print(f'Optimizer: {optimizer_name} with learning rate: {lr} and scheduler: {scheduler_name} \n')
            train_loss, val_loss = train_model(model, train_loader, val_loader, device, optimizer=optimizer, num_epochs=30, scheduler=scheduler)
            # val_accuracies_per_optimizer[optimizer_name] = val_accuracies[-1] # save the last accuracy
            print('-'*50)

            # save the best combination for the lowest validation loss, used later then
            if np.min(val_loss) < best_combi['val_loss']:
                best_combi['val_loss'] = np.min(val_loss)
                best_combi['train_loss'] = np.min(train_loss)
                best_combi['optimizer'] = optimizer_name
                best_combi['scheduler'] = scheduler_name
                best_combi['lr'] = lr
                best_combi['epoch'] = np.argmin(val_loss)

            if lr == 0.001:
                train_val_loss_001.append((train_loss[-1], val_loss[-1]))
            elif lr == 0.005:
                train_val_loss_005.append((train_loss[-1], val_loss[-1]))
            elif lr == 0.01:
                train_val_loss_01.append((train_loss[-1], val_loss[-1]))

            fig,axs = plt.subplots(1,2,layout="tight")
            axs[0].plot(train_loss)
            axs[0].set_title("training loss")
            axs[0].set_xlabel("epoch")
            axs[0].set_ylabel("MSE loss")

            axs[1].plot(val_loss)
            axs[1].set_title("validation loss")
            axs[1].set_xlabel("epoch")
            axs[1].set_ylabel("MSE loss")
            plt.suptitle(f"{optimizer_name}")
            plt.show()
            # reset model and optimizer https://www.youtube.com/watch?v=r9tOQ6EKS1Y&ab_channel=deeplizard
            for layer in model.children():
                layer.reset_parameters()

print(best_combi)


# In[9]:


# print(train_val_loss_001)
# print(train_val_loss_005)
# print(train_val_loss_01)


# ## d) Final Training with best parameters

# In[10]:


# {'optimizer': 'Adam', 'scheduler': 'MultiStepLR', 'lr': 0.005, 'val_loss': np.float64(0.26152706938328535), 'train_loss': np.float64(0.19305862985815095), 'epoch': np.int64(29)} aktuellstes
final_training_model = NeuralNet_deeper_wide()
final_X_train = torch.concatenate((X_train_torch, X_validation_torch))
final_y_train = torch.concatenate((y_train_torch, y_validation_torch))
final_train_dataset = TensorDataset(final_X_train,final_y_train)
final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_seed))

lr = 0.005
# final_X_train.to(device)
# final_y_train.to(device)

final_optimizer = torch.optim.Adam(final_training_model.parameters(), lr=lr)
# final_scheduler = CosineAnnealingWarmRestarts(final_optimizer, T_0=15)
final_scheduler = MultiStepLR(final_optimizer, milestones=[5, 10, 20], gamma=0.5)

train_loss, val_loss = train_model(final_training_model, final_train_loader, device=device, optimizer=final_optimizer, num_epochs=29, scheduler=final_scheduler)
# val_accuracies_per_optimizer[optimizer_name] = val_accuracies[-1] # save the last accuracy
print('-'*50)

# save the best combination for the lowest validation loss, used later then

fig2,axs = plt.subplots(1,1,layout="tight")
axs.plot(train_loss)
axs.set_title("training loss")
axs.set_xlabel("epoch")
axs.set_ylabel("MSE loss")
plt.show()
# fig2.savefig("./plots/final_train.png")


# In[11]:


print(f"final_training_loss = {train_loss[-1]}")


# In[12]:


# Test
final_training_model.to(device) # Set model to gpu
final_training_model.eval()
X_test_torch = X_test_torch.to(device)
y_test_torch = y_test_torch.to(device)
loss_fn = nn.MSELoss()
# Run forward pass
with torch.no_grad():
  pred = final_training_model(X_test_torch)
loss = loss_fn(y_test_torch, pred)

fig3 = plt.figure()
plt.scatter(pred.cpu(), y_test_torch.cpu(), alpha=0.8)
plt.ylabel("Ground Truth")
plt.xlabel("Predicted")
plt.title("Test Set")
plt.show()
print(f"final test loss = {loss}")
# fig3.savefig("./plots/scatter.png")


# ## f) Binary Classification

# In[5]:


# from models import NeuralNet_deeper_wide_classification
from torch.optim.lr_scheduler import MultiStepLR

#combine training data back together with validation data
X_train_norm = normalized_sets[0]
X_val_norm = normalized_sets[1]
X_test_norm = normalized_sets[2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_recombined = np.concatenate((X_train_norm, X_val_norm), axis=0)
y_train_recombined = np.concatenate((y_train, y_validation),axis=0)

y_train_recombined_class = y_train_recombined.copy()
y_test_class = y_test.copy()
y_train_recombined_class[y_train_recombined < 2], y_test_class[y_test < 2] = 0., 0.
y_train_recombined_class[y_train_recombined >= 2], y_test_class[y_test >= 2] = 1., 1.


#
# ##########################################################
#
#
X_train_torch = torch.tensor(X_train_recombined,dtype=torch.float32)
y_train_torch = torch.tensor(y_train_recombined_class.reshape((len(y_train_recombined_class),1)),dtype=torch.float32)
X_test_torch = torch.tensor(X_test_norm,dtype=torch.float32)
y_test_torch = torch.tensor(y_test_class.reshape((len(y_test),1)),dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train_torch,y_train_torch)
test_dataset = torch.utils.data.TensorDataset(X_test_torch,y_test_torch)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_seed))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_seed))
# #########################################################
#
loss_fn = nn.NLLLoss()


#Create the full model(NN, optimizer, scheduler), depending on what performed best
best_learning_rate = 0.005
best_num_epochs = 30
best_model = NeuralNet_deeper_wide_classification().to(device)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate)
best_scheduler = MultiStepLR(best_optimizer, milestones=[5, 10, 20], gamma=0.5)

avg_training_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1,best_num_epochs+1):# one loop over the dataset = 1 epoch
    print('-'*20, f'Epoch {epoch}', '-'*20)
    # Train one epoch
    best_model.train() #> setting model to "train mode", because some layers act different in training or evaluation
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()

        best_optimizer.zero_grad()#needed, because otherwise the new gradients would get summed onto the old ones
        # the sum would be useful e.g. for CNNs

        log_probs = torch.squeeze(best_model(data))# log probabilities of batch
        target = torch.squeeze(target.long())
        loss = loss_fn(log_probs, target)
        loss.backward()
        best_optimizer.step()

        losses.append(loss.item())
        # if batch_idx % 100 == 0:
        #     print(f'Train Epoch {epoch} | Loss: {loss.item()}')
    best_scheduler.step()
    avg_train_loss = np.mean(losses[-len(train_loader):])
    avg_training_losses.append(avg_train_loss)
    print(f'\nAverage train loss in epoch {epoch}: {avg_train_loss}')
    
    # Evaluate on test set
    best_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            log_probs = torch.squeeze(best_model(data))
            target = torch.squeeze(target.long())
            
            test_loss += F.nll_loss(log_probs, target, reduction='sum').item()  # sum up batch loss
            pred = torch.argmax(log_probs, dim=1)  # get the index of the max log-probability
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)
    avg_correct = correct / len(test_loader.dataset)

    test_losses.append(test_loss)
    test_accuracies.append(avg_correct)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * avg_correct:.0f}%)\n')


# In[6]:


fig1,axs = plt.subplots(1,3,layout="tight",figsize = (10,5))
axs[0].plot(np.linspace(1,len(avg_training_losses),len(avg_training_losses)),avg_training_losses)
axs[0].set_title("training loss")
#axs[0].set_ylim([0.28,max(avg_training_losses)])
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("NLL loss")

axs[1].plot(np.linspace(1,len(test_losses),len(test_losses)),test_losses)
axs[1].set_title("test loss")
#axs[1].set_ylim([0.28,max(test_losses)])
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("NLL loss")

axs[2].plot(np.linspace(1,len(test_accuracies),len(test_accuracies)), [100*acc for acc in test_accuracies])
axs[2].set_title("test accuracy")
axs[2].set_ylim([0,100])
axs[2].set_xlabel("epoch")
axs[2].set_ylabel("accuracy in %")
plt.suptitle(f"{best_model._get_name()}")
plt.show()
# fig1.savefig("./plots/binary_classification_test_evaluation.png")


# In[7]:


print(f"final training loss: {avg_training_losses[-1]}")
print(f"final test loss: {test_losses[-1]}")
print(f"final test accuracy: {test_accuracies[-1]}")


# In[ ]:




