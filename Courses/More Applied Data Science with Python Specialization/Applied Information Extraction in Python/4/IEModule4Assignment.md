# Assignment 4: Transformers

This week, you learned about transformers. In this assignment, you will be implementing aspects of a transformer pipeline in python in order to again make predictions on fields of the address dataset.



```python
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
import json
import numpy as np
no_deprecation_warning=True
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

The solution for this code is the same as in assignment 3. 






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

# Call the student solution
stu_X, stu_y = task_1_solution()

print(f"Task {task_id} - your X:\n{stu_X[:2]}")  # Display a subset of the data for clarity
print(f"Task {task_id} - your y:\n{stu_y[:2]}")  # Display a subset of the data for clarity

assert isinstance(stu_X, list), f"Task {task_id}: The returned X should be a list."
assert isinstance(stu_y, list), f"Task {task_id}: The returned y should be a list."

assert len(stu_X) == len(
    stu_y
), f"Task {task_id}: The lengths of X and y should match. Found len(X)={len(stu_X)} and len(y)={len(stu_y)}."

assert all(
    isinstance(seq, list) and all(isinstance(word, str) for word in seq)
    for seq in stu_X
), f"Task {task_id}: Each element of X should be a list of strings."

assert all(
    isinstance(seq, list) and all(isinstance(label, str) for label in seq)
    for seq in stu_y
), f"Task {task_id}: Each element of y should be a list of strings."

```

    Task 1 - AG tests
    Task 1 - your X:
    [['777', 'Brockton', 'Avenue', 'Abington', 'MA', '2351'], ['30', 'Memorial', 'Drive', 'Avon', 'MA', '2322']]
    Task 1 - your y:
    [['address', 'address', 'address', 'city', 'state', 'zip'], ['address', 'address', 'address', 'city', 'state', 'zip']]


# Task 2: Encode Data 

Like with our bidirectional LSTM CRF model, we must encode the data we've read into a format appropriate for our transformer model. You will be implementing a part of this.

The first step has been done for you, loading the tokenizer.

Then, we loop through in the `sequences` and `label` arrays in parallel. 

For each token sequence and label sequence:
 - We define a `tokenized` dictionary with the current sequence as the first argument. We also use ` is_split_into_words=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt"`

 - Now, we should be able to access elements of the `tokenized` dictionary with `[]` and access the "input_ids" and "attention_mask" fields. We call `.squeeze()` on the arrays we obtain when subscripting into the dictionary and append them to `input_ids` and `attention_masks`.

 - The next step is to define label_ids. We create a new label id list variable that we will later append to `label_ids`

- This next part is where you come in! 

 - Still inside of the loop for each sequence and label, we write another loop that loops through the word ids in `tokenized.word_ids()` in the `word_ids` variable, which looks like this: 

 ```[None, 0, 0, 1, 1, 2, 3, 3, 4, 5, 5, None, None, None,.... ]```

- Inside of this subloop: 
  - Check each word id. If it is of type `None`, append the integer -100 to the `label_id_seq` list variable created before this subloop began. 
  - Otherwise it should be an integer you can use to subscript into the current label sequence(`label_seq`) from the outer loop to obtain a label. Use the `state_to_idx` to convert that label into it's corresponding index. Append that to `label_id_seq`. 

- Now, we exit the subloop and append `torch.tensor(('label_id_seq'), dtype=torch.long)` to label_ids. All done!
    
 


```python
task_id = '2'


```


```python
def task_2_solution():
    sequences, labels = task_1_solution()
    state_to_idx = {"address": 0, "city": 1, "state": 2, "zip": 3}


    tokenizer = BertTokenizerFast.from_pretrained("assets/tokenizer_dir")
    
    input_ids, attention_masks, label_ids = [], [], []


    for seq, label_seq in zip(sequences, labels):
        tokenized = tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        input_ids.append(tokenized["input_ids"].squeeze())

        attention_masks.append(tokenized["attention_mask"].squeeze())
        
        word_ids = tokenized.word_ids()    

        label_id_seq = []
        for word_id in word_ids:
            
            if word_id is None:
                label_id_seq.append(-100)
            else:
                label = label_seq[word_id]
                label_id_seq.append(state_to_idx[label])

        label_ids.append(torch.tensor(label_id_seq, dtype=torch.long))
    
    return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(label_ids)



```


```python
# Use this cell to explore your solution.

task_2_solution()
```




    (tensor([[  101,  6255,  2581,  ...,     0,     0,     0],
             [  101,  2382,  3986,  ...,     0,     0,     0],
             [  101,  5539, 13381,  ...,     0,     0,     0],
             ...,
             [  101, 19527,  8482,  ...,     0,     0,     0],
             [  101,  3429, 22025,  ...,     0,     0,     0],
             [  101, 24368,  2629,  ...,     0,     0,     0]]),
     tensor([[1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             ...,
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0]]),
     tensor([[-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             ...,
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100]]))




```python
print(f"Task {task_id} - AG tests")

stu_X, stu_attention_masks, stu_labels = task_2_solution()

print(f"Task {task_id} - your input_ids:\n{stu_X[:2]}")  # Display a subset of the data for clarity
print(f"Task {task_id} - your attention_masks:\n{stu_attention_masks[:2]}")  # Display a subset of the data for clarity
print(f"Task {task_id} - your labels:\n{stu_labels[:2]}")  # Display a subset of the data for clarity

assert isinstance(stu_X, torch.Tensor), f"Task {task_id}: The returned `X` should be a torch.Tensor."
assert isinstance(stu_attention_masks, torch.Tensor), f"Task {task_id}: The returned `attention_masks` should be a torch.Tensor."
assert isinstance(stu_labels, torch.Tensor), f"Task {task_id}: The returned `labels` should be a torch.Tensor."

assert stu_X.shape == stu_attention_masks.shape == stu_labels.shape, f"Task {task_id}: The shapes of `X`, `attention_masks`, and `labels` should match. Found X.shape={stu_X.shape}, attention_masks.shape={stu_attention_masks.shape}, labels.shape={stu_labels.shape}"

assert stu_X.size(1) == 128, f"Task {task_id}: Each sequence in `X` should have 128 tokens. Found a sequence of length {stu_X.size(1)}."
assert stu_attention_masks.size(1) == 128, f"Task {task_id}: Each sequence in `attention_masks` should have 128 tokens. Found a sequence of length {stu_attention_masks.size(1)}."
assert stu_labels.size(1) == 128, f"Task {task_id}: Each sequence in `labels` should have 128 tokens. Found a sequence of length {stu_labels.size(1)}."



```

    Task 2 - AG tests
    Task 2 - your input_ids:
    tensor([[  101,  6255,  2581, 13899,  2669,  3927, 11113,  7853,  5003, 17825,
              2487,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0],
            [  101,  2382,  3986,  3298, 16131,  5003, 20666,  2475,   102,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0]])
    Task 2 - your attention_masks:
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0]])
    Task 2 - your labels:
    tensor([[-100,    0,    0,    0,    0,    0,    1,    1,    2,    3,    3, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100],
            [-100,    0,    0,    0,    1,    2,    3,    3, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100]])


# Task 3: Create Dataset class


Just like with Bidirectional LSTM CRF it helps us to create a dataset class so we can create Dataloaders, which store our data in batches. 

Again, your job is to just define the `__init__` class, where you set `self.input_ids`, `self.attention_masks`, and `self.label_ids` to their corresponding passed in arguments. 

Afterwards, create train and test `AddressDataset` objects from the variables we retrieved from `train_test_split`. For `train_dataset`, `input_ids` should be set to `X_train`, `attention_masks` to `attn_train`, and `label_ids` to `y_train`. You will do the same for `test_dataset` but with the test versions of these variables we have at the beginning.

Finally, create a train loader and test loader object with `Dataloader`, where the `dataset` arguments are the train and test datasets you just defined. Also set `batch_size` to 16.

The return statement has been defined for you.




```python
task_id = '3'

## Utility Code will go here

```


```python
def task_3_solution():
    # Получаем закодированные данные из Task 2
    input_ids, attention_masks, label_ids = task_2_solution()

    # Train / test split
    X_train, X_test, attn_train, attn_test, y_train, y_test = train_test_split(
        input_ids,
        attention_masks,
        label_ids,
        test_size=0.2,
        random_state=42
    )

    # Dataset class
    class AddressDataset(Dataset):
        def __init__(self, input_ids, attention_masks, label_ids):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.label_ids = label_ids

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return (
                self.input_ids[idx],
                self.attention_masks[idx],
                self.label_ids[idx]
            )

    # Create datasets
    train_dataset = AddressDataset(X_train, attn_train, y_train)
    test_dataset = AddressDataset(X_test, attn_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader, input_ids, attention_masks, label_ids
```


```python
# Use this cell to explore your solution.

task_3_solution()
```




    (<torch.utils.data.dataloader.DataLoader at 0x7c0333427f10>,
     <torch.utils.data.dataloader.DataLoader at 0x7c0333427eb0>,
     tensor([[  101,  6255,  2581,  ...,     0,     0,     0],
             [  101,  2382,  3986,  ...,     0,     0,     0],
             [  101,  5539, 13381,  ...,     0,     0,     0],
             ...,
             [  101, 19527,  8482,  ...,     0,     0,     0],
             [  101,  3429, 22025,  ...,     0,     0,     0],
             [  101, 24368,  2629,  ...,     0,     0,     0]]),
     tensor([[1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             ...,
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0],
             [1, 1, 1,  ..., 0, 0, 0]]),
     tensor([[-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             ...,
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100],
             [-100,    0,    0,  ..., -100, -100, -100]]))




```python
print(f"Task {task_id} - AG tests")

# Call the student solution
stu_train_loader, stu_test_loader, stu_input_ids, stu_attention_masks, stu_label_ids = task_3_solution()

# Check return types
assert isinstance(stu_train_loader, DataLoader), f"Task {task_id}: The returned `train_loader` should be a PyTorch DataLoader."
assert isinstance(stu_test_loader, DataLoader), f"Task {task_id}: The returned `test_loader` should be a PyTorch DataLoader."
assert isinstance(stu_input_ids, torch.Tensor), f"Task {task_id}: The returned `input_ids` should be a torch.Tensor."
assert isinstance(stu_attention_masks, torch.Tensor), f"Task {task_id}: The returned `attention_masks` should be a torch.Tensor."
assert isinstance(stu_label_ids, torch.Tensor), f"Task {task_id}: The returned `label_ids` should be a torch.Tensor."



```

    Task 3 - AG tests


# Task 4: Build Model

Here, we're going to define the model and optimizer we will be using for our transformer, which will be returned by the function below.

Set the model as `BertForTokenClassification.from_pretrained()`. The first argument should be to the model we've stored for you in assets/bert_token_classification, and num_labels should be 4.

Optimizer should be `AdamW()`, with the first argument being `model.parameters()` and the learning rate `lr` argument set to the number `5e-5`.



```python
task_id = '4'

## Utility Code will go here

```


```python
def task_4_solution():
    # Load model from local pretrained checkpoint
    model = BertForTokenClassification.from_pretrained(
        "assets/bert_token_classification",
        num_labels=4
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5
    )

    return model, optimizer
```


```python
# Use this cell to explore your solution.

task_4_solution()
```




    (BertForTokenClassification(
       (bert): BertModel(
         (embeddings): BertEmbeddings(
           (word_embeddings): Embedding(30522, 768, padding_idx=0)
           (position_embeddings): Embedding(512, 768)
           (token_type_embeddings): Embedding(2, 768)
           (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
           (dropout): Dropout(p=0.1, inplace=False)
         )
         (encoder): BertEncoder(
           (layer): ModuleList(
             (0-11): 12 x BertLayer(
               (attention): BertAttention(
                 (self): BertSdpaSelfAttention(
                   (query): Linear(in_features=768, out_features=768, bias=True)
                   (key): Linear(in_features=768, out_features=768, bias=True)
                   (value): Linear(in_features=768, out_features=768, bias=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
                 (output): BertSelfOutput(
                   (dense): Linear(in_features=768, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (intermediate): BertIntermediate(
                 (dense): Linear(in_features=768, out_features=3072, bias=True)
                 (intermediate_act_fn): GELUActivation()
               )
               (output): BertOutput(
                 (dense): Linear(in_features=3072, out_features=768, bias=True)
                 (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                 (dropout): Dropout(p=0.1, inplace=False)
               )
             )
           )
         )
       )
       (dropout): Dropout(p=0.1, inplace=False)
       (classifier): Linear(in_features=768, out_features=4, bias=True)
     ),
     AdamW (
     Parameter Group 0
         betas: (0.9, 0.999)
         correct_bias: True
         eps: 1e-06
         lr: 5e-05
         weight_decay: 0.0
     ))




```python
print(f"Task {task_id} - AG tests")

# Call the student solution
stu_model, stu_optimizer = task_4_solution()

# Check return types
assert isinstance(stu_model, BertForTokenClassification), f"Task {task_id}: The returned `model` should be an instance of `BertForTokenClassification`."
assert isinstance(stu_optimizer, AdamW), f"Task {task_id}: The returned `optimizer` should be an instance of `AdamW`."

# Check model configuration
assert stu_model.config.num_labels == 4, f"Task {task_id}: The `num_labels` in the model configuration should be 4. Found {stu_model.config.num_labels}."
assert stu_model.config._name_or_path == "assets/bert_token_classification", f"Task {task_id}: The model should use the pretrained `assets/bert_token_classification`. Found {stu_model.config._name_or_path}."

# Check optimizer configuration
optimizer_params = list(stu_optimizer.param_groups[0].keys())
assert "lr" in optimizer_params, f"Task {task_id}: The optimizer should have a `lr` parameter."
assert stu_optimizer.param_groups[0]["lr"] == 5e-5, f"Task {task_id}: The learning rate for the optimizer should be `5e-5`. Found {stu_optimizer.param_groups[0]['lr']}."



```

    Task 4 - AG tests


# Task 5: Evaluate model

Now, you've conducted the steps necessary to use the transformer to make predictions on the labels of the address dataset. 

The code itself to actually train and simulate the model is similar to the code we've written in the previous module with the Bidirectional-LSTM CRF model. However, this is too computationally expensive to execute on our systems, so we have prepared the outputs of a trained model for you in a precomputed file. Your job here is simply to deduce the accuracy of the outputs and return it in the function below.


To do this, 

We Loop through each batch in `precomputed_predictions`. For each batch you can access "predictions" and "label_ids" of the batch with something like `batch["predictions"]`. We've done all this and converted these to numpy 2d arrays for you. 

Now, your job is to loop through these arrays and find the accuracy of the transformer model. 

- Hint: Padding tokens are stored in `label_ids` ndarrays as the integer `-100`. Make sure to exclude padding when computing accuracy. For example, if we have 

preds = [[0, 0], [0, 0]]

label_ids = [[-100, 1, 0, -100], [-100, 0, 0, -100]]

Accuracy here would be 75%.








```python
task_id = '5'

## Utility Code will go here


```


```python

    
def task_5_solution():
    # Load precomputed predictions
    precomputed_predictions = torch.load("assets/precomputed_predictions.pt")

    correct, total = 0, 0

    # Evaluate predictions
    for batch in precomputed_predictions:
        predictions = batch["predictions"].numpy()
        label_ids = batch["label_ids"].numpy()

        # loop over batch and sequence
        for preds_seq, labels_seq in zip(predictions, label_ids):
            for pred, label in zip(preds_seq, labels_seq):
                # ignore padding
                if label == -100:
                    continue
                if pred == label:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
```


```python
# Use this cell to explore your solution.

task_5_solution()
```




    0.9918533604887984




```python
print(f"Task {task_id} - AG tests")

# Call the student solution
stu_accuracy = task_5_solution()

# Check return type
assert isinstance(stu_accuracy, float), f"Task {task_id}: The returned `accuracy` should be of type `float`. Found {type(stu_accuracy)}."


```

    Task 5 - AG tests


# Task 6

Now, you've computed the accuracy of predicting address fields with CRF, Bidirectional LSTM-CRF, and a Transformer Model. There should be a slight performance difference between each model. In this task, we'd like you to rank the accuracy of the 3 prediction methods you used. 

Using the string variables we've defined for you below, rank the each method in ascending order of accuracy.

Note: The accuracy you obtain from these models can vary from run to run. Models that are more accurate than others may have an unlucky run and come out with a lower accuracy than a less accurate model. If you've ranked them based on a single run of the models and came out with the incorrect answer, try running the models multiple times and rank them based on the average values you see. 


```python
task_id = '6'

## Utility Code will go here
```


```python
def task_6_solution():
    BidirectionalLSTMCRF = "BidirectionalLSTMCRF"
    crf = "crf"
    Transformer = "Transformer"
    
    sol = [crf, BidirectionalLSTMCRF,Transformer]
    # YOUR CODE HERE


    return sol
```


```python
# Use this cell to explore your solution.

task_6_solution()
```




    ['crf', 'BidirectionalLSTMCRF', 'Transformer']




```python
print(f"Task {task_id} - AG tests")
stu_ans = task_6_solution()

print(f"Task {task_id} - your answer:\n{stu_ans}")

assert isinstance(
    stu_ans, list
), f"Task {task_id}: Your function should return a list. "

assert (
    len(stu_ans) == 3
), f"Task {task_id}: Your returned list has an incorrect number of answers. "


assert all(
    isinstance(ans, str) for ans in stu_ans
), (
    f"Task {task_id}: Your returned list should be a list of strs"
)


```

    Task 6 - AG tests
    Task 6 - your answer:
    ['crf', 'BidirectionalLSTMCRF', 'Transformer']



```python

```
