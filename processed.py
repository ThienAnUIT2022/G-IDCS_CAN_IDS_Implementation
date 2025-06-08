import csv
import os
import re

def process_normal_txt(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass'])
        writer.writeheader()
        
        for line in fin:
            # Xử lý định dạng đặc biệt của file txt
            match = re.match(r'Timestamp: (\d+\.\d+).*?ID: (\w+).*?DLC: (\d+)\s+([\w\s]+)$', line)
            if match:
                writer.writerow({
                    'Timestamp': match.group(1),
                    'Arbitration_ID': match.group(2),
                    'DLC': match.group(3),
                    'Data': ' '.join(match.group(4).split()),
                    'Class': 'Normal',
                    'SubClass': 'Normal'
                })

def process_hacking_csv(input_path, output_path, attack_type):
    with open(input_path, 'r') as fin, open(output_path, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.DictWriter(fout, fieldnames=['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass'])
        writer.writeheader()
        
        for row in reader:
            # Xử lý định dạng CSV cho các tấn công
            data_bytes = row[3:-1]  # Lấy các byte dữ liệu
            writer.writerow({
                'Timestamp': row[0],
                'Arbitration_ID': row[1],
                'DLC': row[2],
                'Data': ' '.join(data_bytes),
                'Class': 'Attack' if row[-1] == 'T' else 'Normal',
                'SubClass': attack_type if row[-1] == 'T' else 'Normal'
            })

# Đưỡng dẫn đến các file đầu vào và đầu ra
input_base = 'W:\\IDPS-project-dataset\\BaoCaoCK\\Train_TH'
output_base = 'W:\\IDPS-project-dataset\\BaoCaoCK\\Train_TH'

# Xử lý file normal_run_data.txt
process_normal_txt(
    os.path.join(input_base, 'normal_run_data.txt'),
    os.path.join(output_base, 'normal_processed.csv')
)

# # ----------------------------------------------------------------------
# # Đưỡng dẫn đến các file đầu vào và đầu ra
# input_base = 'W:\\IDPS-project-dataset\\BaoCaoCK\\Test_TH'
# output_base = 'W:\\IDPS-project-dataset\\BaoCaoCK\\Test_TH'

# # Xử lý các file tấn công có định dạng CSV
# attack_mapping = {
#     'DoS_dataset.csv': 'DoS',
#     'Fuzzy_dataset.csv': 'Fuzzy',
#     'gear_dataset.csv': 'Gear_Spoofing',
#     'RPM_dataset.csv': 'RPM_Spoofing'
# }

# for fname, attack_type in attack_mapping.items():
#     process_hacking_csv(
#         os.path.join(input_base, fname),
#         os.path.join(output_base, f'{attack_type}_processed.csv'),
#         attack_type
#     )
