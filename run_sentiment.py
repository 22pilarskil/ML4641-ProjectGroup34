import torch

def train_model(epoch, model, loader, loss_fn, optimizer, device):

    total_loss = 0

    for batch_num, batch in enumerate(loader):

        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.max(dim=1)[1]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")



def evaluate_model(model, loader, loss_fn, device):

    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.max(dim=1)[1] 

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predicted_labels == labels).item()

    return total_loss / len(loader), correct_predictions / len(loader.dataset)
