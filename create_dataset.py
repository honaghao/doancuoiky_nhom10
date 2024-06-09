import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands #Tạo một đối tượng để sử dụng mediapipe cho việc nhận diện bàn tay.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) #Khởi tạo bộ nhận diện bàn tay: chế độ ảnh tĩnh, số bàn tay tối đa, Ngưỡng tin cậy tối thiểu để xác định một bàn tay trong hình ảnh.
#Khi thuật toán nhận diện một bàn tay trong hình ảnh, nó sẽ đưa ra một giá trị tin cậy (confidence score) cho việc xác định đó. Nếu giá trị tin cậy này thấp hơn min_detection_confidence, thì thuật toán sẽ coi nhận diện bàn tay là không chính xác và sẽ không trả về kết quả. Ở đây giá trị là 0.3 (trên thang 0 - 1)


DATA_DIR = './data' #Đường dẫn đến thư mục chứa hình ảnh

# Số lượng landmarks và số chiều (x, y) cho mỗi landmark
NUM_LANDMARKS = 21 #Số lượng của các điểm đặc trưng (landmarks)
NUM_DIMENSIONS = 2 #số chiều của nó (2 chiều x,y)

data = [] #Khởi tạo danh sách trống để lưu trữ dữ liệu và nhãn.
labels = []
for dir_ in os.listdir(DATA_DIR): #Lặp qua các thư mục con trong thư mục dữ liệu
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #Lặp qua các hình ảnh trong từng thư mục
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) #đọc hình ảnh từ đường dẫn được xác định
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Chuyển đổi hình ảnh từ màu BGR (màu mặc định trong OpneCV) sang RGB
        results = hands.process(img_rgb) #xử lý hình ảnh để xác định các landmarks của bàn tay
        
        # Khởi tạo mảng với số lượng features cố định
        data_aux = np.zeros(NUM_LANDMARKS * NUM_DIMENSIONS, dtype=float) 
        
        if results.multi_hand_landmarks: #Kiểm tra xem có landmarks của bàn tay được tìm thấy trong hình ảnh hay không
            hand_landmarks = results.multi_hand_landmarks[0] #Lấy các landmarks của bàn tay đầu tiên nếu có.
            for i, landmark in enumerate(hand_landmarks.landmark): #Lặp qua các landmarks của bàn tay 
                # Lấy chỉ số tương ứng trong mảng data_aux
                idx = i * NUM_DIMENSIONS
                data_aux[idx] = landmark.x #Lưu tọa độ x và y của landmarks vào mảng data_aux.
                data_aux[idx + 1] = landmark.y
        
        data.append(data_aux) #Thêm data_aux và nhãn vào danh sách dữ liệu và nhãn.
        labels.append(dir_)

# Lưu dữ liệu và nhãn vào file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f) #Tập tin pickle này chứa một từ điển với hai khóa 'data' và 'labels' tương ứng với dữ liệu và nhãn đã thu thập.
