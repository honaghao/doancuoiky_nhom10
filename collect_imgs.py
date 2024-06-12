########################################Train lại với 27 data################################################


#Dùng thư viện OpenCV2 (cv2) để thu thập dữ liệu từ camera

import os

import cv2


DATA_DIR = './data' #dẫn tới thư mực sẽ lưu dữ liệu
if not os.path.exists(DATA_DIR): #kiểm tra xem đã có thư mục chưa, nếu chưa thì tạo
    os.makedirs(DATA_DIR)

number_of_classes = 27 #số lượng lớp (hoặc danh mục) muốn tạo (27 ký tự)
dataset_size = 800 #số hình ảnh muốn thu thập cho mỗi lớp (càng lớn thì mô hình càng chính xác)

cap = cv2.VideoCapture(0) #kết nối tới camera (camera 0 là mặc định của laptop)
for j in range(number_of_classes): #lặp qua từng lớp
    if not os.path.exists(os.path.join(DATA_DIR, str(j))): #kiểm tra xem có tồn tại chưa, nếu chưa thì tự tạo
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True: #vòng lặp vô hạn được dùng để thu thập dữ liệu bằng cách nhấn 'Q'
        ret, frame = cap.read()  #đọc một khung hình từ camera
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, #vị trí (100,50), màu (0, 255, 0)
                    cv2.LINE_AA)
        cv2.imshow('frame', frame) #dùng để hiển thị khung hình với văn bản 'Ready? Press "Q" ! :)
        if cv2.waitKey(25) == ord('q'): #nếu nhấn 'Q' thì bắt đầu thoát khỏi vòng lặp và thu thập dữ liệu
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read() #đọc khung hình từ camera và lưu vào biến frame
        cv2.imshow('frame', frame) 
        cv2.waitKey(25) #chờ 25 milliseconds trước khi chuyển tới khung hình tiếp theo
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) #lưu khung hình hiện tại vào tập tin ảnh trong thư mục tương ứng với lớp và với tên là số thứ tự của ảnh

        counter += 1 #tăng giá trị biến đếm lên đến khi bằng data_size thì kết thúc

cap.release() #giải phóng tài nguyên camera để không bị đầy (hoặc tràn) bộ nhớ
cv2.destroyAllWindows() #đóng toàn bộ cửa sổ