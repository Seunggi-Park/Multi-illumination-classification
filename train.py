
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import Dataloader as DR
import Model_res as MD
import Classifier as Classifier
from model_zoo import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
def label_smoothing(labels, num_classes=5, smoothing=0.2):
    confidence = 1.0 - smoothing
    smooth_label = torch.full(size=(labels.size(0), num_classes), fill_value=smoothing / (num_classes - 1)).to(device)
    smooth_label.scatter_(1, labels.data.unsqueeze(1), confidence)
    return smooth_label


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleClassifier, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(512 * 6 * 6, 512)  # Flattening the input and connecting to fully connected layers
        self.fc2 = nn.Linear(512, num_classes)  # Output layer with the number of classes

    def forward(self, x):
        x = x.view(-1, 512 * 6 * 6)  # Flatten the input tensor
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.fc2(x)  # Output layer (logits)
        return x
train_dataset = DR.CustomDatasetOver(root_dir='dataset/train', transform=DR.transform2)
valid_dataset = DR.CustomDataset(root_dir='dataset/valid', transform=DR.transform2)
test_dataset = DR.CustomDataset(root_dir='dataset/test', transform=DR.transform2)

#train_dataset2 = DR.CustomDatasetOver(root_dir='dataset/train', transform=DR.transform2)
print(len(train_dataset))
#train_loader2 = DataLoader(train_dataset2, batch_size=8, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 훈련 설정
baseline = SqueezeNet1_0()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier.MVClassifier_Proposed(baseline,numclasses=5)  # 클래스 개수로 변경
model = model.to(device)
sc1 = SimpleClassifier(num_classes=5).to(device)
sc2 = SimpleClassifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()
optimizer_AE = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

class_list = ['BrightLine', 'Deformation', 'Dent', 'Normal', 'Scratch']
# 훈련 및 검증 루프
num_epochs = 120

patience = 10
early_stopping_counter = 0
best_valid_accuracy = 0.0
best_model_path = 'proposed_ae_2_no_smothing.pth'
total_params = MD.count_parameters(model)
print(best_model_path)
print(f"Total trainable parameters: {total_params}")
best_train_loss = float('inf')  # 초기값을 무한대로 설정

best_valid_accuracy = 0.0
best_test_accuracy = 0.0# 초기값 설정
best_train_loss = float('inf')  # 초기값을 무한대로 설정
for epoch in range(50):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = [img.to(device) for img in images]
        labels = [class_list.index(label) for label in labels]
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        smoothed_labels = label_smoothing(labels, smoothing=0.4)
        losses = 0.0
        logits, fusion_logit, x1, x2, decoded1, decoded2, encoded1, encoded2 = model(images, 5)
        # for logit in logits:
        #     logit = logit.unsqueeze(0) if logit.dim() == 1 else logit
        #     losses += criterion(logit, labels)
        losses += criterion2(x1, decoded2) + criterion2(x2, decoded1) + 0.1*criterion(sc1(encoded1),smoothed_labels) + 0.1*criterion(sc2(encoded2),smoothed_labels)

        losses.backward()
        optimizer_AE.step()

        running_loss += losses.item()
    epoch_loss = running_loss / len(train_loader)
    print("AEloss",epoch_loss)
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
# for epoch in range(20):
#     # Training phase
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images = [img.to(device) for img in images]
#         labels = [class_list.index(label) for label in labels]
#         labels = torch.tensor(labels).to(device)
#         optimizer.zero_grad()
#         smoothed_labels = label_smoothing(labels, smoothing=0.0)
#         losses = 0.0
#         logits, fusion_logit, x1, x2, decoded1, decoded2, encoded1, encoded2 = model(images, 5)
#         # for logit in logits:
#         #     logit = logit.unsqueeze(0) if logit.dim() == 1 else logit
#         #     losses += criterion(logit, labels)
#         losses += 0.1*criterion(sc1(encoded1),smoothed_labels) + 0.1*criterion(sc2(encoded2),smoothed_labels)
#
#         losses.backward()
#         optimizer_AE.step()
#
#         running_loss += losses.item()
#     epoch_loss = running_loss / len(train_loader)
#     print("AEloss",epoch_loss)
#     model.eval()
#     valid_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = [img.to(device) for img in images]
#             labels = [class_list.index(label) for label in labels]
#             labels = torch.tensor(labels).to(device)
#             loses = 0
#             logits, fusion_logit, _, _, _, _, e1, e2 = model(images, 5)
#             smoothed_labels = label_smoothing(labels)
#
#             fusion_logit = sc2(e2)
#
#             valid_loss += losses.item()
#             _, predicted = torch.max(fusion_logit, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         valid_loss = valid_loss / len(test_loader)
#         valid_accuracy = 100 * correct / total
#         print(valid_accuracy)
#     valid_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = [img.to(device) for img in images]
#             labels = [class_list.index(label) for label in labels]
#             labels = torch.tensor(labels).to(device)
#             loses = 0
#             logits, fusion_logit, _, _, _, _, e1, e2 = model(images, 5)
#             smoothed_labels = label_smoothing(labels)
#
#
#             fusion_logit = sc1(e1)
#
#             valid_loss += losses.item()
#             _, predicted = torch.max(fusion_logit, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         valid_loss = valid_loss / len(test_loader)
#         valid_accuracy = 100 * correct / total
#         print(valid_accuracy)
#
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = [img.to(device) for img in images]
        labels = [class_list.index(label) for label in labels]
        labels = torch.tensor(labels).to(device)
        smoothed_labels = label_smoothing(labels, smoothing=0.0)
        smoothed_labels2= label_smoothing(labels, smoothing=0.0)
        optimizer.zero_grad()
        losses = 0.0
        logits, fusion_logit, x1, x2, decoded1, decoded2, _, _ = model(images, 5)
        for logit in logits:
            logit = logit.unsqueeze(0) if logit.dim() == 1 else logit
            losses += criterion(logit, smoothed_labels)

        fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit
        losses += (criterion(fusion_logit, smoothed_labels2) + criterion2(x1, decoded2) + criterion2(x2, decoded1))

        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    epoch_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = [img.to(device) for img in images]
            labels = [class_list.index(label) for label in labels]
            labels = torch.tensor(labels).to(device)
            loses = 0
            logits, fusion_logit, _, _, _, _, _, _= model(images, 5)
            smoothed_labels = label_smoothing(labels)
            for logit in logits:
                logit = logit.unsqueeze(0) if logit.dim() == 1 else logit
                losses += criterion(logit, smoothed_labels)

            fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit

            valid_loss += losses.item()
            _, predicted = torch.max(fusion_logit, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = valid_loss / len(valid_loader)
    valid_accuracy = 100 * correct / total

    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images = [img.to(device) for img in images]
            labels = [class_list.index(label) for label in labels]
            labels = torch.tensor(labels).to(device)
            smoothed_labels = label_smoothing(labels)

            logits, fusion_logit, _, _, _, _, _, _ = model(images, 5)
            fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit
            _, predicted = torch.max(fusion_logit, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    train_accuracy = 100 * train_correct / train_total

    with torch.no_grad():
        for images, labels in test_loader:
            images = [img.to(device) for img in images]
            labels = [class_list.index(label) for label in labels]
            labels = torch.tensor(labels).to(device)

            logits, fusion_logit, _, _, _, _, _, _ = model(images, 5)
            fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit
            _, predicted = torch.max(fusion_logit, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    if test_accuracy >= best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_test_accuracy.pth')
    if valid_accuracy >= best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved with accuracy: {valid_accuracy:.2f}%')

    if epoch_loss < best_train_loss:
        best_train_loss = epoch_loss
        early_stopping_counter = 0  # Reset the counter if we get a new best train loss
    else:
        early_stopping_counter += 1

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%, train accuracy: {train_accuracy:.2f}% test accuracy: {test_accuracy:.2f}%')

    # Early stopping
    if early_stopping_counter >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs')
        break

print('Training complete')

# Best model 로드
model.load_state_dict(torch.load(best_model_path))

# Test phase
model.eval()
test_loss = 0.0
correct = 0
total = 0


correct_classwise = {i: 0 for i in range(len(class_list))}
total_classwise = {i: 0 for i in range(len(class_list))}


with torch.no_grad():
    for images, labels in test_loader:
        images = [img.to(device) for img in images]
        labels = [class_list.index(label) for label in labels]
        labels = torch.tensor(labels).to(device)

        logits, fusion_logit, _, _, _, _, _, _= model(images, 5)
        fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit
        _, predicted = torch.max(fusion_logit, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update class-wise correct predictions and total samples
        for label, prediction in zip(labels, predicted):
            total_classwise[label.item()] += 1
            if label == prediction:
                correct_classwise[label.item()] += 1

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Calculate and print class-wise accuracy
for i, class_name in enumerate(class_list):
    if total_classwise[i] > 0:
        accuracy = 100 * correct_classwise[i] / total_classwise[i]
        print(f'Class {class_name} Accuracy: {accuracy:.2f}%')
    else:
        print(f'Class {class_name} Accuracy: N/A (no samples)')
