import torch
import torchvision
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import math

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb
def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
  # Use ImageFolder to create dataset(s)
  train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
  test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
  
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  
  test_dataloader = torch.utils.data.DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/06_pytorch_transfer_learning_exercise_solutions.ipynb
def get_initial_model(num_classes, device):
    model = torchvision.models.efficientnet_b0(pretrained=True).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=num_classes, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    return model

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb
def train_step(model, dataloader, loss_fn, optimizer, device):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb
def test_step(model, dataloader, loss_fn,device):
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# SOURCE: Demetrius Gulewicz
def get_test_data(NN_test_dir, num_img, test_data_dir):
    valid_test = np.loadtxt(test_data_dir, delimiter =',')
    all_idxs = valid_test
    transform1 = torchvision.transforms.ToTensor()
    X_test = torch.empty(num_img,3,224,224)
    
    prev_idx = -1
    
    for i in range(num_img):
        idx = int(all_idxs[i])
        
        if idx == prev_idx:
            counter = counter + 1
        else:
            counter = 0
            
        prev_idx = idx
        
        X_test[i,:,:,:] = transform1(Image.open(NN_test_dir + str(idx) + '_' + str(counter) + '.jpg'))
        
    return X_test

# SOURCE: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/06_pytorch_transfer_learning_exercise_solutions.ipynb
def make_itemized_prediction(model_0, X_test, num_test_img):
    num_iter = math.ceil(num_test_img / 64)

    # Make predictions on the entire test dataset
    test_preds = []
    test_probs = []
    model_0.eval()
    counter = 0

    with torch.inference_mode():
        for i in range(num_iter):
            # Pass the data through the model
            counter_max = min(counter + 64,num_test_img)
            test_logits = model_0(X_test[counter:counter_max,:,:,:])
            counter = counter + 64
            
            # Convert the pred logits to pred probs
            pred_probs = torch.softmax(test_logits, dim=1)

            # Convert the pred probs into pred labels
            pred_labels = torch.argmax(pred_probs, dim=1)
            
            # add pred probs to test probs list
            test_probs.append(torch.max(pred_probs,dim=1).values)

            # Add the pred labels to test preds list
            test_preds.append(pred_labels)

    # Concatenate the test preds and put them on the CPU
    test_preds = torch.cat(test_preds).cpu()
    test_probs = torch.cat(test_probs).cpu()
    
    return test_preds, test_probs
        
# SOURCE: Demetrius Gulewicz
def get_aggregate_prediction(test_preds, test_probs, test_data_dir):
    # get dup number
    dup_number = np.loadtxt(test_data_dir, delimiter =',')

    # filtered predictions
    filt_pred = []
    work_idx = 0

    # all base image indexes
    j_max = int(max(dup_number)) + 1

    for j in range(j_max):
        idxs = np.where(dup_number == j)[0]
        num_idxs = len(idxs)
        
        if num_idxs == 1:
            filt_pred.append(test_preds[work_idx].item())
            work_idx = work_idx + 1
        else:
            max_idx = torch.argmax(test_probs[work_idx:work_idx+num_idxs])
            filt_pred.append(test_preds[work_idx+max_idx].item())
            work_idx = work_idx + num_idxs
    
    return filt_pred, dup_number, j_max