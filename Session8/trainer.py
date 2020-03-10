from tqdm import tqdm
import torch


def fit(train_loader,
        test_loader,
        model,
        n_epochs,
        optimizer,
        criterion,
        device,
        l1_reg=False,
        l1_lambda=0,
        return_misclassified=True,
        number_of_images_to_return=25,
        return_worst_predictions=True):

    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []

    misclassified_dict_args = dict(return_misclassified=return_misclassified, 
                              number_of_images_to_return=number_of_images_to_return,
                              return_worst_predictions=return_worst_predictions)

    for epoch in range(n_epochs):
        print("EPOCH:", epoch)
        train_epoch_acc, train_epoch_losses = train_epoch(model,
                                                          device,
                                                          train_loader,
                                                          optimizer,
                                                          epoch,
                                                          l1_reg,
                                                          l1_lambda=l1_lambda)
        train_acc.extend(train_epoch_acc)
        train_losses.extend(train_epoch_losses)

        if epoch == n_epochs-1:
            test_acc_epoch, test_losses_epoch, misclassified_list = test_epoch(model,
                                                                               device,
                                                                               test_loader,
                                                                               ** misclassified_dict_args)
        else:    
            test_acc_epoch, test_losses_epoch = test_epoch(model, device, test_loader)

        test_acc.append(test_acc_epoch)
        test_losses.append(test_losses_epoch)

        if return_misclassified:
            return train_acc, train_losses, test_acc, test_losses, misclassified_list
        else:
            return train_acc, train_losses, test_acc, test_losses

def train_epoch(model,
                device,
                train_loader,
                optimizer,
                criterion,
                epoch,
                l1_reg=False,
                l1_lambda=0):

  train_epoch_acc = []
  train_epoch_loss = []
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)

    loss = criterion(y_pred, target)

    if l1_reg:    
        l1_crit = nn.L1Loss(size_average=False)
        reg_loss = 0
        for param in model.parameters():
            zero_tensor = torch.zeros(param.size()).to(device)
            reg_loss += l1_crit(param, zero_tensor)
      
        loss += l1_lambda * reg_loss

    loss.backward()
    train_epoch_loss.append(loss.item())
    optimizer.step()
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_epoch_acc.append(100*correct/processed)

  return train_epoch_acc, train_epoch_loss

def test_epoch(model, 
               device,
               test_loader, 
               criterion,
               return_misclassified=False, 
               number_of_images_to_return=0, 
               return_worst_predictions=False):

    misclassified_images_list = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_value = torch.max(output, dim=1).values

            
            match_tensor = pred.eq(target.view_as(pred))
            misclassified_images_in_batch = torch.where(match_tensor == False)
            for eachimage in misclassified_images_in_batch[0]:

               image_predlist = [data[eachimage], pred[eachimage].item(), pred_value[eachimage].item(), target[eachimage].item()]
               misclassified_images_list.append(image_predlist)
            correct += match_tensor.sum().item()

    test_loss /= len(test_loader.dataset)
    

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc = 100. * correct / len(test_loader.dataset)

    if return_misclassified:
      if not return_worst_predictions:
        return test_acc, test_loss, misclassified_images_list[:number_of_images_to_return]
      else:
        sorted_on_pred_value = sorted(misclassified_images_list, key = lambda x: x[2], reverse=True)
        return test_acc, test_loss, sorted_on_pred_value[:number_of_images_to_return]
    
    return test_acc, test_loss