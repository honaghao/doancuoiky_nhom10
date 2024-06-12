import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Mở và tải dữ liệu từ tệp pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Chuyển data và labels thành mảng numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Chia data thành tập training (80%) và tập testing (20%)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Tạo danh sách để lưu các giá trị dưới
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

# Số lần chia nhỏ data
n_splits = 50
# Kích thước mỗi chunk của data training
chunk_size = len(x_train) // n_splits

for i in range(1, n_splits + 1):
    # Get subset of training data
    x_train_subset = x_train[:i * chunk_size] # Lấy subset của data training
    y_train_subset = y_train[:i * chunk_size] # Lấy subset của labels training

    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Khởi tạo model RandomForest
    model.fit(x_train_subset, y_train_subset)  # Train model trên tập con

    # Predict on the training set
    y_train_predict = model.predict(x_train_subset) # Dự đoán trên tập training
    y_train_predict_proba = model.predict_proba(x_train_subset) # Dự đoán xác suất trên tập training

    # Predict on the test set
    y_test_predict = model.predict(x_test) # Dự đoán trên tập testing
    y_test_predict_proba = model.predict_proba(x_test) # Dự đoán xác suất trên tập testing

    # Calculate training accuracy and log loss
    train_acc = accuracy_score(y_train_subset, y_train_predict) # Tính train accuracy
    train_loss = log_loss(y_train_subset, y_train_predict_proba) # Tính train loss

    # Calculate validation accuracy and log loss
    val_acc = accuracy_score(y_test, y_test_predict) # Tính validation accuracy
    val_loss = log_loss(y_test, y_test_predict_proba) # Tính validation loss

    # Lưu các giá trị accuracy and loss
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    val_losses.append(val_loss)

    # In train accuracy và train loss cho từng epoch hiện tại
    print(f'Epoch {i}: Train Accuracy = {train_acc}, Train Loss = {train_loss}, Val Accuracy = {val_acc}, Val Loss = {val_loss}')

# Plot accuracy
epochs = range(1, n_splits + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy over epochs')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss over epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Train final model on full training data
final_model = RandomForestClassifier(n_estimators=100, random_state=42) # Khởi tạo final model RandomForest
final_model.fit(x_train, y_train) # Train final model trên toàn bộ data training

# Predict on the test set
final_y_predict = final_model.predict(x_test) # Dự đoán trên tập testing

# Calculate final accuracy
final_score = accuracy_score(final_y_predict, y_test) # Tính final accuracy

print('{}% of samples were classified correctly!'.format(final_score * 100)) # In tỷ lệ mẫu được phân loại chính xác

# Save the final model to file
with open('model.p', 'wb') as f:
    pickle.dump({'model': final_model}, f) # Lưu final model vào tệp pickle
