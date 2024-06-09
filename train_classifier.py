import pickle
from sklearn.ensemble import RandomForestClassifier #tạo và huấn luyện mô hình RandomForest
from sklearn.model_selection import train_test_split #chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.metrics import accuracy_score #để tính toán độ chính xác của mô hình.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss





# Load the data from the pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Dung để kiểm tra cấu trúc và kiểu dữ liệu của dữ liệu đã tải từ tập tin pickle, giúp kiểm tra và chẩn đoán vấn đề nếu có.
print(f"Type of data_dict['data']: {type(data_dict['data'])}")
print(f"Type of elements in data_dict['data']: {type(data_dict['data'][0])}")
print(f"First element in data_dict['data']: {data_dict['data'][0]}")
print(f"Shape of first element in data_dict['data']: {np.array(data_dict['data'][0]).shape}")

#Sử dụng vòng lặp để tìm chiều dài lớn nhất của các chuỗi trong dữ liệu.
max_len = max(len(d) for d in data_dict['data'])

# Ensure that all data points are the same length by padding them,  các giá trị 0 được thêm vào cuối chuỗi cho đến khi độ dài của chuỗi đạt đến độ dài lớn nhất của các chuỗi trong tập dữ liệu
data = np.array([np.pad(d, (0, max_len - len(d)), 'constant', constant_values=0) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Print shapes to confirm consistency
print(f"Shape of data: {data.shape}")
print(f"Shape of labels: {labels.shape}")

# Check if all data points have the same shape
if len(set(d.shape for d in data)) != 1:
    raise ValueError("Not all data points have the same shape")

# Chia dữ liệu thành hai phần: tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model
model = RandomForestClassifier()

# Lists to store loss and accuracy values
train_losses = []
train_accuracies = []

# Number of epochs
num_epochs = 10  # Thay đổi số lượng epochs tùy ý

# Train the model
for epoch in range(num_epochs):
    model.fit(x_train, y_train)
    # Calculate training loss
    y_pred_train = model.predict_proba(x_train)
    train_loss = log_loss(y_train, y_pred_train)
    # Calculate initial training loss
    if epoch == 0:
        train_loss_initial = train_loss
    # Convert loss to percentage
    train_loss_percentage = (train_loss / train_loss_initial) * 100
    train_losses.append(train_loss_percentage)
    # Calculate training accuracy
    train_accuracy = model.score(x_train, y_train)
    train_accuracies.append(train_accuracy)

# Number of epochs
epochs = len(train_losses)

# Biểu đồ loss
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (%)')
plt.legend()
plt.show()

# Biểu đồ accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()