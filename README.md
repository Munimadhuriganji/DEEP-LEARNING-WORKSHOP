# DEEP-LEARNING-WORKSHOP
## NAME:GANJI MUNI MADHURI
## REFERENCE NO:212223230060
# PROGRAM:
```
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

df = pd.read_csv('income.csv')
print(len(df))
df.head()
df['label'].value_counts()
df.columns
cat_cols = ['sex','education','marital-status','workclass','occupation']
cont_cols = ['age','hours-per-week']
y_col = ['label']
print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
for col in cat_cols:
    df[col] = df[col].astype('category')
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats[:5]
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]
conts = torch.tensor(conts, dtype=torch.float32)
conts.dtype
y = torch.tensor(df[y_col].values).flatten()
b = 30000   # batch size
t = 5000    # test size

cat_train = cats[:b-t]
cat_test  = cats[b-t:b]
con_train = conts[:b-t]
con_test  = conts[b-t:b]
y_train   = y[:b-t]
y_test    = y[b-t:b]
class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()
        
        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        # Assign a variable to hold a list of layers
        layerlist = []
        
        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)
        
        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        # Set up model layers
        x = self.layers(x)
        return x
torch.manual_seed(33)
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import time
import torch
start_time = time.time()

epochs = 300
losses = []

for i in range(1, epochs+1):
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss Curve")
plt.show()
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)

print(f'CE Loss: {loss.item():.8f}')
correct = 0
for i in range(len(y_test)):
    if y_val[i].argmax().item() == y_test[i]:
        correct += 1

acc = correct / len(y_test)
print(f'{correct} out of {len(y_test)} = {acc*100:.2f}% correct')

```
# OUTPUT:
<img width="892" height="425" alt="image" src="https://github.com/user-attachments/assets/65a5ecbe-bf21-43a0-a6f3-3d7dc41bc9d7" />

<img width="242" height="133" alt="image" src="https://github.com/user-attachments/assets/a9a0e6ce-5682-49ad-8995-4d96ab1ec9a5" />

<img width="648" height="106" alt="image" src="https://github.com/user-attachments/assets/f975c04f-f6f1-4e91-a311-a51d96f810b4" />

<img width="238" height="85" alt="image" src="https://github.com/user-attachments/assets/a974e4d5-f13f-4e47-a4e7-0b338147a42b" />

<img width="904" height="336" alt="image" src="https://github.com/user-attachments/assets/f5c17d7b-a048-4266-a95e-ae3a67be6226" />

<img width="385" height="171" alt="image" src="https://github.com/user-attachments/assets/468528fd-feff-4f0b-9f47-a32e2b5b012f" />

<img width="187" height="167" alt="image" src="https://github.com/user-attachments/assets/ec79dd58-42fd-4a7d-8329-8d85d7b5cf34" />

<img width="153" height="70" alt="image" src="https://github.com/user-attachments/assets/f7067df9-9f66-41d8-b83f-b2393c131958" />

<img width="369" height="78" alt="image" src="https://github.com/user-attachments/assets/5c364978-2791-4271-8fe5-7fc53002f84f" />

<img width="853" height="415" alt="image" src="https://github.com/user-attachments/assets/d5e5a11f-58c9-4419-a8f7-1abb20fdfbd2" />



<img width="274" height="324" alt="image" src="https://github.com/user-attachments/assets/b400cfec-91dc-4dd6-9321-5f4b17eecfa1" />



<img width="730" height="564" alt="image" src="https://github.com/user-attachments/assets/461f27e9-c9c7-4eaf-8d1b-77d0b6f07628" />



<img width="180" height="30" alt="image" src="https://github.com/user-attachments/assets/720b347d-dc63-4ca4-9e4f-a985f8639d3a" />


<img width="308" height="27" alt="image" src="https://github.com/user-attachments/assets/c4168536-092c-4096-b6cb-263969301d27" />






























