import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
from parallelattention import TimeSeriesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils_window import TimeSeriesScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torchsummary import summary
torch.manual_seed(0) #set seed for reproducibility
np.random.seed(0)



data = np.load('mimic4_acute_resp_failure_vsdata_ldcbf_4209_048_19_btf.npz')
data_all, label = data['arr_0'], data['arr_1'].reshape(-1,1)
data_scaler = TimeSeriesScaler()
data_x = data_scaler.fit_transform(data_all)


# print(torch.from_numpy(data_all).shape, torch.from_numpy(label).shape)

X = torch.from_numpy(data_x)
y = torch.from_numpy(label.reshape(-1,1))
# y = torch.from_numpy(label)


train_x, test_x, train_y, test_y= train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)


# Data Loaders
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(test_x, test_y)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# Hyperparameters
num_features = 19  # Adjust this to the number of features in your dataset
num_classes = 1   # Adjust this to the number of classes in your dataset
seg_len = 48       # Segment number or 
#time steps seg_num = 48
# factor = 8         # Factor for the router
d_model = 16       # Model dimension
n_heads = 2       # Number of attention heads

# Instantiate model
model = TimeSeriesClassifier(num_features, num_classes, seg_len, n_heads)
# Print the summary

# print(model.summary())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('Running on',device)

model.to(device) 

# summary(model, input_size=(seg_len, num_features))


# Loss and Optimizer
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


### implement LR reducer, increase training accuracy


# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Initialize lists to store predictions and labels for each batch
    train_preds = []
    train_labels = []
    train_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.float()  # Ensure batch_X is in float32 format

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        # outputs = model(batch_X).squeeze()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.float())
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        
        probs = torch.sigmoid(outputs)  # Convert logits to probabilities
        preds = (probs >= 0.5).float()  # Threshold probabilities to generate predictions

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(batch_y.cpu().numpy())

        # print(train_preds[0].shape, train_labels[0].shape)
        # break
    # break

    # Train metrics

    train_accuracy = accuracy_score(train_labels, train_preds)
    # train_precision = precision_score(train_labels, train_preds, average='weighted')
    # train_recall = recall_score(train_labels, train_preds, average='weighted')
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    train_mcc= matthews_corrcoef(train_labels, train_preds)
    
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    # Validation step
    model.eval()  # Set model to evaluation mode
    val_preds = []
    val_labels = []
    val_losses = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.float()
            batch_y = batch_y.float()

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # outputs = model(batch_X).squeeze()
            outputs = model(batch_X)

            val_loss = criterion(outputs, batch_y)
            val_losses.append(val_loss.item()) 


            # print(outputs)
            # Apply sigmoid and threshold at 0.5 for predictions
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())


    # Validation metrics
    val_accuracy = accuracy_score(val_labels, val_preds)
    # val_precision = precision_score(val_labels, val_preds, average='weighted')
    # val_recall = recall_score(val_labels, val_preds, average='weighted')
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_mcc= matthews_corrcoef(val_labels, val_preds)


    print(f"Epoch {epoch+1}/{num_epochs}, "
    	  f"Train Loss: {np.mean(train_losses):.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}, Train MCC: {train_mcc:.4f}, "
          f"Val Loss:   {np.mean(val_losses):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}, Val MCC: {val_mcc:.4f}")


# def extract_attention_weights(model):
#     attention_weights = []
#     for layer in model.children():
#         if isinstance(layer, FullAttention):
#             attention_weights.append(layer.attention_weights)
#         else:
#             attention_weights.extend(extract_attention_weights(layer))
#     return attention_weights

# # Run a forward pass
# output = model(input_data)

# # Extract attention weights
# attention_weights_list = extract_attention_weights(model)