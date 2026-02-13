# Assignment 3: Bidirectional LSTM CRF Model

This week, you learned about the bidirectional LSTM and CRF models. In this assignment, you will see how we can implement these models in python.



```python
import json
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np

# Load the dataset
with open("assets/list_of_real_usa_addresses.json", "r") as file:
    addresses_json = json.load(file)

# Define constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


```

# Task 1: Set up data

In this task, you will be responsible for formatting the dataset into the arrays X and y, where X is a list of addresses split up into tokens, and y is a list containing the corresponding labels. 

Ex. For the following data:

`[{
    "address": "123 Main St",
    "city": "Springfield",
    "state": "IL",
    "zip": "62704"
}, etc.]`

The function should create X and y like so:


`X = [["123", "Main", "St", "Springfield", "IL", "62704"], etc.]`


`y = [["address", "address", "address", "city", "state", "zip"], etc.]`








```python
task_id = '1'

## Utility Code will go here



```


```python
def task_1_solution():
    X = []
    y = []


    one_x = []
    one_y = []
    for i in addresses_json:

        for k in i.keys():
            
            one_x = one_x + i[k].split(' ')
            one_y = one_y + [k] * len(i[k].split(' '))

        X.append(one_x)
        y.append(one_y)

        one_x = []
        one_y = []
        

    return X, y
```


```python
print(f"Task {task_id} - AG tests")

stu_X, stu_y = task_1_solution()

print(f"Task {task_id} - your X:\n{stu_X[:2]}")  # Display a subset of the data for clarity
print(f"Task {task_id} - your y:\n{stu_y[:2]}")  # Display a subset of the data for clarity

assert isinstance(stu_X, list), f"Task {task_id}: The returned `X` should be a list."
assert isinstance(stu_y, list), f"Task {task_id}: The returned `y` should be a list."

assert len(stu_X) == len(
    stu_y
), f"Task {task_id}: The lengths of `X` and `y` should match. Found len(X)={len(stu_X)} and len(y)={len(stu_y)}."

assert all(
    isinstance(seq, list) and all(isinstance(word, str) for word in seq)
    for seq in stu_X
), f"Task {task_id}: Each element of `X` should be a list of strings."

assert all(
    isinstance(seq, list) and all(isinstance(label, str) for label in seq)
    for seq in stu_y
), f"Task {task_id}: Each element of `y` should be a list of strings."


```

    Task 1 - AG tests
    Task 1 - your X:
    [['777', 'Brockton', 'Avenue', 'Abington', 'MA', '2351'], ['30', 'Memorial', 'Drive', 'Avon', 'MA', '2322']]
    Task 1 - your y:
    [['address', 'address', 'address', 'city', 'state', 'zip'], ['address', 'address', 'address', 'city', 'state', 'zip']]


# Task 2: Create encodings for data:

To move further, we need to create a vocabulary and a label-to-index mapping strings to integer represntations to prepare the data for further processing.

The label-to-index mapping has been created for you, and the first elements of the address token vocabulary have been set for you. 

In the function below, you need to populate `vocab` with the rest of the tokens seen in the X array from task_1. 

For example, for an X like: 

`[["123", "Main", "St", "Springfield", "IL", "62704"], ["124", "Main", "St", "Springfield", "IL", "62705"], etc.]`

Your vocab would look like:

`{
    "<PAD>": 0,
    "<UNK>": 1,
    "123": 2,
    "Main": 3,
    "St": 4,
    "Springfield": 5,
    "IL": 6,
    "62704": 7,
    "124" : 8,
    "62705" : 9,
    etc.
}`

The order for assigning indices in vocab should be in order of appearance in X.


```python
task_id = '2'



```


```python
def task_2_solution():
    X, _ = task_1_solution()

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    state_to_idx = {PAD_TOKEN: 0, "address": 1, "city": 2, "state": 3, "zip": 4}

    num = 2

    for i in X:

        for k in i:

            if k not in list(vocab.keys()):

                vocab[k] = num
                num = num + 1

    return vocab, state_to_idx


```


```python
print(f"Task {task_id} - AG tests")

stu_vocab, _ = task_2_solution()

print(f"Task {task_id} - your vocab (first 10 entries):\n{list(stu_vocab.items())[:10]}")

assert isinstance(
    stu_vocab, dict
), f"Task {task_id}: The returned `vocab` should be a dictionary."




```

    Task 2 - AG tests
    Task 2 - your vocab (first 10 entries):
    [('<PAD>', 0), ('<UNK>', 1), ('777', 2), ('Brockton', 3), ('Avenue', 4), ('Abington', 5), ('MA', 6), ('2351', 7), ('30', 8), ('Memorial', 9)]


# Task 3: Convert Sequences to Indices and Find Longest Sequence

Now, use the maps we created in task 2 to convert the sequences and labels datasets we created in task 1 from datasets of strings to being their corresponding integer representations in the map. 

For example, if you have 


`[["123", "Main", "St", "Springfield", "IL", "62704"], ["124", "Main", "St", "Springfield", "IL", "62705"], etc.]`

and a vocab of:

`{
    "<PAD>": 0,
    "<UNK>": 1,
    "123": 2,
    "Main": 3,
    "St": 4,
    "Springfield": 5,
    "IL": 6,
    "62704": 7,
    "124" : 8,
    "62705" : 9,
    etc.
}`


X should be converted to:

`X = [[2, 3, 4, 5, 6, 7], [8, 3, 4, 5, 6, 9], etc.]`

Do the same for `labels` using `state_to_idx` to create `y`

Now is also a good time to find the longest sequence of tokens in your sequences array. For example, the example entry we have above in X is 6 tokens long, and if this was the longest sequence in our dataset, we'd return 6 as an integer. Find the longest sequence and return it in the integer `max_length`.



```python
task_id = '3'

## Utility Code will go here

    

```


```python
def task_3_solution():
    sequences, labels = task_1_solution()
    vocab, state_to_idx = task_2_solution()

    X = []
    y = []
    max_length = 0
    one_x = []
    one_y = []
    for i, j in zip(sequences, labels):

        for z,n in zip(i,j):

            one_x.append(vocab[z])
            one_y.append(state_to_idx[n])

        X.append(one_x)
        one_x = []
        y.append(one_y)
        one_y = []

        if len(j)>max_length:

            max_length = len(j)

        
            

        

    return X, y, max_length

```


```python
print(f"Task {task_id} - AG tests")

stu_X, stu_y, stu_max_length = task_3_solution()

print(f"Task {task_id} - your tokenized sequences (first 2):\n{stu_X[:2]}")
print(f"Task {task_id} - your label indices (first 2):\n{stu_y[:2]}")
print(f"Task {task_id} - your max sequence length: {stu_max_length}")

assert isinstance(
    stu_X, list
), f"Task {task_id}: The returned `X` should be a list of tokenized sequences."
assert isinstance(
    stu_y, list
), f"Task {task_id}: The returned `y` should be a list of label index sequences."
assert isinstance(
    stu_max_length, int
), f"Task {task_id}: The returned `max_length` should be an integer."

assert len(stu_X) == len(
    stu_y
), f"Task {task_id}: The number of tokenized sequences in `X` does not match the number of label sequences in `y`."

for seq in stu_X:
    assert isinstance(seq, list), f"Task {task_id}: Each sequence in `X` should be a list."
    assert all(
        isinstance(token, int) for token in seq
    ), f"Task {task_id}: Each token in `X` should be an integer."
for seq in stu_y:
    assert isinstance(seq, list), f"Task {task_id}: Each sequence in `y` should be a list."
    assert all(
        isinstance(label, int) for label in seq
    ), f"Task {task_id}: Each label in `y` should be an integer."

assert all(
    len(seq) <= stu_max_length for seq in stu_X
), f"Task {task_id}: All sequences in `X` should have lengths less than or equal to `max_length`."
assert all(
    len(seq) <= stu_max_length for seq in stu_y
), f"Task {task_id}: All sequences in `y` should have lengths less than or equal to `max_length`."



```

    Task 3 - AG tests
    Task 3 - your tokenized sequences (first 2):
    [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 6, 12]]
    Task 3 - your label indices (first 2):
    [[1, 1, 1, 2, 3, 4], [1, 1, 1, 2, 3, 4]]
    Task 3 - your max sequence length: 9


# Task 4: Create Dataset class

Now, we need to wrap the data we processed in the previous tasks into dataset objects.

This function will return two instances of these objects, being a train and test dataset. The return statement has been defined for you. 

Your job here is to complete the `__init__` function, which will assign padded versions of X and y to `self.X` and `self.y`. For example:

For X like:

`X = [[2, 3, 4], [5, 6]]`

If `max_length` is 5, we would assign 

```
[[2, 3, 4, 0, 0],  # Padded to length 5`
[5, 6, 0, 0, 0]]
```
to `self.X`. Recall that `vocab[PAD_TOKEN] = 0`

Do the same with `self.y`






```python
task_id = '4'

## Utility Code will go here


```


```python
def task_4_solution():
    X, y, max_length = task_3_solution()
    vocab, state_to_idx = task_2_solution()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    class AddressDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

            for i in range(len(self.X)):

                X[i] = X[i] + [vocab[PAD_TOKEN]]*(max_length - len(X[i]))
                y[i] = y[i] + [vocab[PAD_TOKEN]]*(max_length - len(y[i]))
            self.X = X
            self.y = y
            
            # This code should come after your code adding the padding
            self.X = torch.tensor(self.X, dtype=torch.long)
            self.y = torch.tensor(self.y, dtype=torch.long)
            

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = AddressDataset(X_train, y_train)
    test_dataset = AddressDataset(X_test, y_test)

    return train_dataset, test_dataset
```


```python
# Use this cell to explore your solution.

task_4_solution()
```




    (<__main__.task_4_solution.<locals>.AddressDataset at 0x7329524b4d60>,
     <__main__.task_4_solution.<locals>.AddressDataset at 0x7329524b6290>)




```python
print(f"Task {task_id} - AG tests")

# Call the student's solution
train_dataset, test_dataset = task_4_solution()

assert isinstance(
    train_dataset, Dataset
), f"Task {task_id}: `train_dataset` should be an instance of `torch.utils.data.Dataset`."
assert isinstance(
    test_dataset, Dataset
), f"Task {task_id}: `test_dataset` should be an instance of `torch.utils.data.Dataset`."

assert len(train_dataset) > 0, f"Task {task_id}: `train_dataset` should not be empty."
assert len(test_dataset) > 0, f"Task {task_id}: `test_dataset` should not be empty."

train_sample_X, train_sample_y = train_dataset[0]  # First training sample
assert isinstance(
    train_sample_X, torch.Tensor
), f"Task {task_id}: Features from `train_dataset` should be a torch.Tensor."
assert isinstance(
    train_sample_y, torch.Tensor
), f"Task {task_id}: Labels from `train_dataset` should be a torch.Tensor."

assert len(train_sample_X) == len(
    train_sample_y
), f"Task {task_id}: The feature and label tensors should have the same length after padding."

```

    Task 4 - AG tests


# Task 5: Build Model


In this task, you will configure the BiLSTM-CRF model.

As you can see, we've defined some starting parameters for you. 

Your job here is to define the `__init__` and `forward` methods.

Inside of the init function, we will be defining the layers of the model:

- create `self.embedding` with `nn.embedding`, with num_embeddings set as the vocab_size, embedding_dim equal to the given embedding_dim, and padding_idx being provided as the arguments. 

- create `self.lstm` with nn.LSTM, with `embedding_dim` and `hidden_dim` as the first two arguments with `batch_first` and `bidirectional` set to `True`.

- create `hidden2label` with nn.Linear, with the `in_features` argument set to twice the value of hidden_dim, and out_features as `num_labels`

- create `self.crf` with `CRF()`, with the `num_tags` argument as `num_labels` and `batch_first` set as True. 

We have defined the `forward` function for you, which is used when training the model and making predictions.


```python
task_id = '5'

## Utility Code will go here

```


```python
def task_5_solution():
    vocab, state_to_idx = task_2_solution()

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    num_labels = len(state_to_idx)
    pad_idx = vocab[PAD_TOKEN]

    
    class BiLSTMCRF(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx):
            super(BiLSTMCRF, self).__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)
            self.crf = CRF(num_labels, batch_first=True)
            

        def forward(self, x, y=None, mask=None):
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            emissions = self.hidden2label(lstm_out)
            if y is not None:  # Training
                return -self.crf(emissions, y, mask=mask)
            else:  # Prediction
                return self.crf.decode(emissions, mask=mask)



    model = BiLSTMCRF(vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx)
    

    return model

```


```python
# Use this cell to explore your solution.

task_5_solution()
```




    BiLSTMCRF(
      (embedding): Embedding(909, 100, padding_idx=0)
      (lstm): LSTM(100, 128, batch_first=True, bidirectional=True)
      (hidden2label): Linear(in_features=256, out_features=5, bias=True)
      (crf): CRF(num_tags=5)
    )




```python
print(f"Task {task_id} - AG tests")

stu_ans = task_5_solution()

assert isinstance(
    stu_ans, nn.Module
), f"Task {task_id}: The returned `model` should be an instance of `torch.nn.Module`."

assert hasattr(
    stu_ans, "embedding"
), f"Task {task_id}: The model should have an attribute `embedding`."
assert isinstance(
    stu_ans.embedding, nn.Embedding
), f"Task {task_id}: The `embedding` attribute should be an instance of `torch.nn.Embedding`."
assert (
    stu_ans.embedding.padding_idx == 0
), f"Task {task_id}: The `embedding` layer should use `PAD_TOKEN` as its `padding_idx`."

assert hasattr(
    stu_ans, "lstm"
), f"Task {task_id}: The model should have an attribute `lstm`."
assert isinstance(
    stu_ans.lstm, nn.LSTM
), f"Task {task_id}: The `lstm` attribute should be an instance of `torch.nn.LSTM`."
assert stu_ans.lstm.bidirectional, f"Task {task_id}: The `lstm` layer should be bidirectional."
assert (
    stu_ans.lstm.input_size == 100
), f"Task {task_id}: The `lstm` layer should have an input size equal to `embedding_dim`."
assert (
    stu_ans.lstm.hidden_size == 128
), f"Task {task_id}: The `lstm` layer should have a hidden size equal to `hidden_dim`."

assert hasattr(
    stu_ans, "hidden2label"
), f"Task {task_id}: The model should have an attribute `hidden2label`."
assert isinstance(
    stu_ans.hidden2label, nn.Linear
), f"Task {task_id}: The `hidden2label` attribute should be an instance of `torch.nn.Linear`."
assert (
    stu_ans.hidden2label.in_features == 256
), f"Task {task_id}: The `hidden2label` layer should have input features equal to `hidden_dim * 2`."

assert hasattr(
    stu_ans, "crf"
), f"Task {task_id}: The model should have an attribute `crf`."
assert isinstance(
    stu_ans.crf, CRF
), f"Task {task_id}: The `crf` attribute should be an instance of `torchcrf.CRF`."


```

    Task 5 - AG tests


# Task 6: Train Model and calculate epoch loss

In this task, you will train the BiLSTM-CRF model you created in the previous task.

Some of the prerequisites have been set for you, but your job will be to implement the loop which trains the model and calculates the loss for each epoch. 

You should loop through each epoch value, in our case (0, 1, 2.... 9). Inside of this loop, you will calculate the epoch loss for each epoch value. Now, for each epoch value:


- Iterate over the X and y batches from the DataLoader we defined for you, `train_loader`. While iterating through each X_batch(batch in X), and y_batch(batch in y)

    - Create a boolean variable that will serve as our mask for this batch which is `true` if X_batch is not equal to model.embedding.padding_idx, and false otherwise.

    - Compute the CRF `loss` object by passing the input (X_batch), true labels (y_batch), and mask to the model (`loss = model(X_batch, ...)`)

    - call `.zero_grad()` on the optimizer object we created for you to zero the gradients. 

    - backpropogate the losses by calling `.backward()` on the loss object. 

    - call `.step()` on the optimizer to update the parameters. 

    - You now have the epoch loss value stored in `loss.item()` This will accumulate as you iterate through train_loader, and you will append the final value to `epoch_losses` each time you leave the train_loader loop to iterate through the next epoch value.

    - make sure to zero the epoch loss after iterating through each epoch value.


```python
task_id = '6'

## Utility Code will go here

```


```python

    
def task_6_solution():
    train_dataset, _ = task_4_solution()
    model = task_5_solution()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:

            mask = X_batch != model.embedding.padding_idx
            loss = model(X_batch, y_batch, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)
    return model, epoch_losses

```


```python
# Use this cell to explore your solution.

task_6_solution()
```




    (BiLSTMCRF(
       (embedding): Embedding(909, 100, padding_idx=0)
       (lstm): LSTM(100, 128, batch_first=True, bidirectional=True)
       (hidden2label): Linear(in_features=256, out_features=5, bias=True)
       (crf): CRF(num_tags=5)
     ),
     [1912.9060668945312,
      1679.1268768310547,
      1383.3875122070312,
      984.0041961669922,
      669.9679412841797,
      414.6939125061035,
      238.70269012451172,
      138.0139217376709,
      82.50282096862793,
      53.6062068939209])




```python
print(f"Task {task_id} - AG tests")
stu_model, stu_epoch_losses = task_6_solution()

print(f"Task {task_id} - Your epoch losses:\n{stu_epoch_losses}")

assert isinstance(
    stu_model, nn.Module
), f"Task {task_id}: Your function should return a PyTorch model (nn.Module)."


assert isinstance(
    stu_epoch_losses, list
), f"Task {task_id}: Your function should return a list of epoch losses as the second output."

assert len(
    stu_epoch_losses
) == 10, f"Task {task_id}: Your returned list of epoch losses should have 10 elements (one for each epoch)."

assert all(
    isinstance(loss, float) for loss in stu_epoch_losses
), f"Task {task_id}: All elements in your epoch losses list should be of type float."

assert all(
    stu_epoch_losses[i] >= stu_epoch_losses[i + 1]
    for i in range(len(stu_epoch_losses) - 1)
), f"Task {task_id}: Your epoch losses should generally decrease over time. Make sure you are appending your values to the end of the return array after each iteration. "

```

    Task 6 - AG tests
    Task 6 - Your epoch losses:
    [1839.9643859863281, 1599.9287567138672, 1299.2947082519531, 901.9044570922852, 571.8404083251953, 335.76806259155273, 198.9085578918457, 117.31461906433105, 74.15816307067871, 49.382076263427734]


# Task 7: Evaluate Model

In this task, you will evaluate the BiLSTM-CRF model you trained in the previous task.

This is similar to the last task: Some of the prerequisites have been set for you, but your job will be to implement the loop which evaluates the model and calculates it's accuracy score. 

In the loop, 

- Iterate through `X_batch, y_batch` from the DataLoader we defined for you, `test_loader`.

    - Create a mask for padding tokens just like you did in the previous part.

    - Compute `preds` by passing the input (X_batch) and mask to the model (`preds = model(X_batch, ...)`). This is the same thing you assigned to the `loss` variable in the previous step.

    - You now have the predicted labels for this batch of addresses in preds, and the actual labels in y_batch. in the form of a a list of lists.
    
    You now have enough information to write code computing the accuracy of the model. Code is given to you which encourages finding the total number of tokens and the total number of correct tokens and calculating accuracy this way, but you may do this any way you wish.

Accuracy should look like this. Let's say you have 

`preds = [[1, 1, 2, 3], [1, 1, 2, 3], [1, 1, 2, 3]]`

and

`y_batch =  [[1, 1, 3, 4, 0], [1, 1, 3, 4, 0], [1, 1, 3, 4, 0]]`

Accuracy would be 50%. 

REMEMBER: `preds` does not contain padding, while `y_batch` does.


```python
task_id = '7'

## Utility Code will go here

_, test_dataset = task_4_solution()
model, _ = task_6_solution()  # Call training to retrieve the trained model
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
model.eval()


```




    BiLSTMCRF(
      (embedding): Embedding(909, 100, padding_idx=0)
      (lstm): LSTM(100, 128, batch_first=True, bidirectional=True)
      (hidden2label): Linear(in_features=256, out_features=5, bias=True)
      (crf): CRF(num_tags=5)
    )




```python
def task_7_solution():

    total_labels = 0
    correct_labels = 0

    with torch.no_grad():  # evaluation = без градиентов
        for X_batch, y_batch in test_loader:
            
            y_batch = y_batch.tolist()

            
            mask = X_batch != model.embedding.padding_idx

            
            preds = model(X_batch, None, mask)

          
            for pred_seq, true_seq in zip(preds, y_batch):
                
                true_seq = true_seq[:len(pred_seq)]

                for p, t in zip(pred_seq, true_seq):
                    total_labels += 1
                    if p == t:
                        correct_labels += 1

    accuracy = correct_labels / total_labels if total_labels > 0 else 0
    return accuracy

```


```python
# Use this cell to explore your solution.

task_7_solution()
```




    0.9936908517350158




```python
print(f"Task {task_id} - AG tests")
stu_accuracy = task_7_solution()

print(f"Task {task_id} - Your accuracy: {stu_accuracy:.4f}")

assert isinstance(stu_accuracy, float), f"Task {task_id}: Your accuracy should be a float."
assert 0 <= stu_accuracy <= 1, f"Task {task_id}: Your accuracy should be between 0 and 1."



```

    Task 7 - AG tests
    Task 7 - Your accuracy: 0.9937

