import os
import json
import numpy as np
from sklearn.metrics import classification_report
from joblib import load
import csv
import networkx as nx

#######################
# Các hàm xử lý dữ liệu và xây dựng đồ thị
#######################

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
    subclass = row.get('SubClass', 'Normal').strip()
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
        'graph_elapsed_time': elapsed_time,
        'max_degree': max_degree,
        'num_edges': num_edges
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
            "subclass": subclass,
            "graph": graph_to_dict(G)    
        }
        windows_results.append(window_result)
    return windows_results

def export_windows_to_json(windows_results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(windows_results, f, indent=4)
    print(f"Window features đã được lưu vào file: {output_file}")

def graph_to_dict(G):
    """
    Chuyển đối tượng đồ thị networkx thành dict có "nodes" và "edges".
    """
    return {
        "nodes": list(G.nodes()),
        "edges": [list(edge) for edge in G.edges()]
    }

#######################
# TH_classifier (Threshold-based)
#######################

def th_classifier(window_feat, thresholds):
    """
    Phân loại window: nếu bất kỳ đặc trưng nào vượt ngoài khoảng normal thì dự đoán "Attack".
    """
    for key, thresh in thresholds.items():
        value = window_feat["features"][key]
        if value < thresh["lower"] or \
            value > thresh["upper"]:
            return "Attack"
    return "Normal"

#######################
# ML_classifier (Machine Learning-based)
#######################

def ml_classifier(window_feat, model):
    """
    Dự đoán nhãn của window dựa trên mô hình ML (Random Forest).
    """
    feat = window_feat["features"]
    X_new = np.array([[feat["graph_elapsed_time"], 
                       feat["max_degree"], 
                       feat["num_edges"]]])
    return model.predict(X_new)[0]

#######################
# Hàm tính metrics và lưu kết quả
#######################

def evaluate_th_classifier(windows, thresholds):
    """
    Đánh giá TH_classifier trên tập windows.
    """
    true_labels = [w["window_label"] for w in windows]
    pred_labels = [th_classifier(w, thresholds) for w in windows]
    report = classification_report(true_labels, pred_labels, output_dict=True)
    return report

def evaluate_ml_classifier(windows, model):
    """
    Đánh giá ML_classifier trên tập windows.
    """
    true_labels = []
    pred_labels = []
    for w in windows:
        label = w["window_label"]
        if label.lower() != "normal":
            label = w["subclass"]
        true_labels.append(label)
        pred_labels.append(ml_classifier(w, model))
    report = classification_report(true_labels, pred_labels, output_dict=True)
    return report

def export_metrics(metrics_dict, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics đã được lưu vào file: {output_file}")

#######################
# Main pipeline
#######################

def main():
    # Đường dẫn file thực nghiệm cho TH_classifier
    input_dir_th = r"W:\\IDPS-project-dataset\\BaoCaoCK\\Test_TH"
    output_windows_dir_th = r"W:\\IDPS-project-dataset\\BaoCaoCK\\windows\\Test_TH"
    th_metrics_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\metrics\\Test_TH_metrics.json"

    # Đường dẫn file thực nghiệm cho ML_classifier
    input_dir_ml = r"W:\\IDPS-project-dataset\\BaoCaoCK\\Test_ML"
    output_windows_dir_ml = r"W:\\IDPS-project-dataset\\BaoCaoCK\\windows\\Test_ML"
    ml_metrics_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\metrics\\Test_ML_metrics.json"

    # Đường dẫn ngưỡng và mô hình
    thresholds_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\th_classifier.json"
    model_path = r"W:\\IDPS-project-dataset\\BaoCaoCK\\ml_classifier.joblib"

    # Kiểm tra tồn tại
    for path in [thresholds_path, model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Tải ngưỡng và mô hình
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)
    clf = load(model_path)

    # ----- TH_classifier: xử lý từng file riêng và xuất windows JSON -----
    os.makedirs(output_windows_dir_th, exist_ok=True)
    th_results = {}
    for file in os.listdir(input_dir_th):
        if file.endswith('_processed.csv'):
            name = os.path.splitext(file)[0]  # e.g. 'DoS_processed'
            path = os.path.join(input_dir_th, file)
            print(f"Processing TH file: {path}")
            windows = process_csv_to_windows(path)
            # lưu windows riêng cho từng file
            out_json = os.path.join(output_windows_dir_th, f"{name}_windows_graphs.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(windows, f, indent=4)
            print(f"Saved TH windows to: {out_json}")

            # đánh giá metrics per file
            report = evaluate_th_classifier(windows, thresholds)
            th_results[name] = report

    # lưu tổng hợp TH metrics
    os.makedirs(os.path.dirname(th_metrics_path), exist_ok=True)
    export_metrics(th_results, th_metrics_path)

     # ----- ML_classifier: xử lý tất cả files chung và xuất windows/Metrics -----
    # os.makedirs(output_windows_dir_ml, exist_ok=True)
    # windows_ml = []
    # for file in os.listdir(input_dir_ml):
    #     if file.endswith('.csv'):
    #         path = os.path.join(input_dir_ml, file)
    #         windows_ml.extend(process_csv_to_windows(path))
    # if not windows_ml:
    #     raise ValueError("No ML windows processed")
    # # xuất windows ML chung
    # out_json_ml = os.path.join(output_windows_dir_ml, 'all_ml_windows.json')
    # with open(out_json_ml, 'w', encoding='utf-8') as f:
    #     json.dump(windows_ml, f, indent=4)
    # print(f"Saved ML windows to: {out_json_ml}")
    # # đánh giá ML metrics
    # ml_report = evaluate_ml_classifier(windows_ml, clf)
    # os.makedirs(os.path.dirname(ml_metrics_path), exist_ok=True)
    # export_metrics(ml_report, ml_metrics_path)

    # # In báo cáo tổng quan
    # print("TH_classifier metrics per file:")
    # print(json.dumps(th_results, indent=4))
    # print("ML_classifier metrics:")
    # print(json.dumps(ml_report, indent=4))

if __name__ == "__main__":
    main()