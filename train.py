import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt
import csv
import networkx as nx

# Các hàm xử lý dữ liệu và xây dựng đồ thị
def parse_csv_row(row):
    """
    Parse một dòng trong CSV có header:
    Timestamp, Arbitration_ID, DLC, Data, Class, SubClass
    """
    try:
        timestamp = float(row.get('Timestamp', row.get('timestamp', None)))
    except:
        timestamp = None
    id = row.get('Arbitration_ID', row.get('ID', '')).strip()
    dlc = int(row.get('DLC', 0)) if row.get('DLC') else None
    data = row.get('Data', row.get('data', '')).split() if row.get('Data', row.get('data')) else []
    msg_class = row.get('Class', 'Normal').strip()
    subclass = row.get('SubClass', 'Normal').strip()  # Mặc định là Normal nếu thiếu SubClass
    return {
        'timestamp': timestamp,
        'ID': id,
        'DLC': dlc,
        'data': data,
        'Class': msg_class,
        'SubClass': subclass
    }

def read_csv_file(file_path):
    messages = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            msg = parse_csv_row(row)
            if msg['timestamp'] is not None:
                messages.append(msg)
    return messages

def build_window_graph(window_msgs):
    """
    Xây dựng đồ thị từ cửa sổ tin nhắn:
     - Các nút: Arbitration_ID
     - Mỗi cạnh: từ tin nhắn hiện tại sang tin nhắn kế tiếp.
    """
    G = nx.DiGraph()
    n = len(window_msgs)
    if n == 0:
        return G
    for i in range(n):
        node = window_msgs[i]['ID']
        if not G.has_node(node):
            G.add_node(node)
        if i < n - 1:
            next_node = window_msgs[i+1]['ID']
            G.add_edge(node, next_node)
    return G

def extract_graph_features(G, window_msgs):
    """
    Trích xuất 3 đặc trưng:
     - graph_elapsed_time: hiệu số giữa timestamp cuối và đầu của cửa sổ.
     - max_degree: bậc lớn nhất (in_degree + out_degree) của đồ thị.
     - num_edges: số lượng cạnh của đồ thị.
    """
    if not window_msgs:
        return None
    elapsed_time = window_msgs[-1]['timestamp'] - window_msgs[0]['timestamp']
    num_edges = G.number_of_edges()
    max_degree = 0
    for node in G.nodes():
        degree = G.in_degree(node) + G.out_degree(node)
        max_degree = max(max_degree, degree)
    return {
        'graph_elapsed_time': elapsed_time, 'max_degree': max_degree, 'num_edges': num_edges
    }

def process_csv_to_windows(file_path, window_size=200):
    """
    Đọc file CSV, chia thành cửa sổ với kích thước window_size, 
    xây dựng đồ thị, trích xuất đặc trưng, và lưu nhãn.
    """
    messages = read_csv_file(file_path)
    windows_results = []
    for i in range(0, len(messages), window_size):
        window_msgs = messages[i:i+window_size]
        if len(window_msgs) < 10:  # Bỏ qua cửa sổ quá nhỏ
            continue
        G = build_window_graph(window_msgs)
        features = extract_graph_features(G, window_msgs)
        label = "Normal"
        subclass = "Normal"
        for msg in window_msgs:
            if msg['Class'].lower() != "normal":
                label = "Attack"
                subclass = msg['SubClass']
                break
        window_result = {
            "window_index": i // window_size + 1, 
            "features": features,
            "window_label": label,
            "subclass": subclass
        }
        windows_results.append(window_result)
    return windows_results

def export_windows_to_json(windows_results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(windows_results, f, indent=4)
    print(f"Window features đã được lưu vào file: {output_file}")

# TH_classifier (Threshold-based)
def compute_thresholds(normal_windows, epsilon=0.03):
    """
    Tính ngưỡng cho từng đặc trưng dựa trên tập windows bình thường.
    """
    times = [w["features"]["graph_elapsed_time"] for w in normal_windows]
    degrees = [w["features"]["max_degree"] for w in normal_windows]
    edges = [w["features"]["num_edges"] for w in normal_windows]
    
    thresholds = {
        "graph_elapsed_time": {
            "lower": min(times) - epsilon * min(times),
            "upper": max(times) + epsilon * max(times)
        },
        "max_degree": {
            "lower": min(degrees) - epsilon * min(degrees),
            "upper": max(degrees) + epsilon * max(degrees)
        },
        "num_edges": {
            "lower": min(edges) - epsilon * min(edges),
            "upper": max(edges) + epsilon * max(edges)
        }
    }
    return thresholds

# Main pipeline
def main():
    # Đường dẫn cho TH_classifier
    input_dir_th = r"W:\\IDPS-project-dataset\\BaoCaoCK\\Train_TH"
    output_json_th = r"W:\\IDPS-project-dataset\\BaoCaoCK\\windows\\train_th_windows.json"
    thresholds_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\th_classifier.json"

    # Đường dẫn cho ML_classifier
    input_dir_ml = r"W:\\IDPS-project-dataset\\BaoCaoCK\\Train_ML"
    output_json_ml = r"W:\\IDPS-project-dataset\\BaoCaoCK\\windows\\train_ml_windows.json"
    model_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\ml_classifier.joblib"

    # ----- TH_classifier processing -----
    windows_th = []
    for file in os.listdir(input_dir_th):
        if file.endswith('.csv'):
            path = os.path.join(input_dir_th, file)
            print(f"Processing TH file: {path}")
            windows_th.extend(process_csv_to_windows(path))
    if not windows_th:
        raise ValueError("No TH windows processed.")
    
    # Xuất TH windows
    with open(output_json_th, 'w', encoding='utf-8') as f:
        json.dump(windows_th, f, indent=4)
    print(f"TH windows saved to: {output_json_th}")

    # Tính thresholds từ TH windows
    normal_windows = [w for w in windows_th if w['window_label'].lower() == 'normal']
    if not normal_windows:
        raise ValueError("No normal windows for TH thresholds.")
   
    thresholds = compute_thresholds(normal_windows, epsilon=0.03)
   
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
    print(f"Thresholds saved to: {thresholds_path}")

    # ----- ML_classifier processing -----
    windows_ml = []
    for file in os.listdir(input_dir_ml):
        if file.endswith('.csv'):
            path = os.path.join(input_dir_ml, file)
            print(f"Processing ML file: {path}")
            windows_ml.extend(process_csv_to_windows(path))
    if not windows_ml:
        raise ValueError("No ML windows processed.")
    
    # Xuất ML windows  
    with open(output_json_ml, 'w', encoding='utf-8') as f:
        json.dump(windows_ml, f, indent=4)
    print(f"ML windows saved to: {output_json_ml}")

    # Chuẩn bị dữ liệu ML
    X = []
    y = []
    for w in windows_ml:
        feat = w['features']
        X.append([feat['graph_elapsed_time'], feat['max_degree'], feat['num_edges']])
        label = w['window_label']
        if label.lower() != 'normal':
            label = w['subclass']
        y.append(label)
    X = np.array(X)

    # Chia train/test và huấn luyện
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    clf = RandomForestClassifier(
        class_weight={'Normal':1,'Flooding':1,'Fuzzing':1,'Replay':1.5,'Spoofing':1.5},
        n_estimators=500, max_depth=10, min_samples_leaf=3, max_features='log2', random_state=42
    )
    clf.fit(X_train, y_train)

    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dump(clf, model_path)
    print(f"Trained and saved ML model to: {model_path}")

if __name__ == "__main__":
    main()