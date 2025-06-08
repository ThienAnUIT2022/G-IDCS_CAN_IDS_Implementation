import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def plot_window_by_index(json_path, window_index):
    """
    Load windows from JSON and plot the graph for the given window_index.
    
    Parameters:
    - json_path: path to the JSON file with windows list
    - window_index: integer index (1-based) of the window to plot
    """
    # Load data
    with open(json_path, 'r') as f:
        windows = json.load(f)
    
    # Find the window with matching index
    window = next((w for w in windows if w['window_index'] == window_index), None)
    if window is None:
        raise ValueError(f"No window found with window_index={window_index}")
    
    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(window['graph']['nodes'])
    G.add_edges_from(window['graph']['edges'])
    
    # Layout and plotting
    pos = nx.spring_layout(G, seed=42)  # fixed seed for reproducibility
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=315)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Graph for Window {window_index}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_window_by_index('windows\\Test_TH\\RPM_Spoofing_processed_windows_graphs.json', window_index=33)

#################################################################
# Load TH_classifier metrics
with open('metrics\\Test_TH_metrics.json', 'r') as f:
    th_metrics = json.load(f)

# Prepare DataFrame for TH (macro avg and accuracy)
th_rows = {}
for name, report in th_metrics.items():
    # strip suffix for readability
    key = name.replace('_processed', '')
    mac = report['macro avg']
    th_rows[key] = {
        'Accuracy': report['accuracy'],
        'Precision': mac['precision'],
        'Recall': mac['recall'],
        'F1-score': mac['f1-score']
    }

df_th = pd.DataFrame.from_dict(th_rows, orient='index')

# Plot grouped bar chart for TH_classifier
ax1 = df_th.plot(kind='bar', figsize=(8, 5))
ax1.set_title("TH_classifier Performance by Attack Type")
ax1.set_ylabel("Score")
ax1.set_ylim(0.9, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()

# export to png
plt.savefig('metrics\\TH_classifier_performance.png', dpi=300)

#################################################################
# Load ML_classifier metrics
with open('metrics\\Test_ML_metrics.json', 'r') as f:
    ml_metrics = json.load(f)

# Classes to plot
classes = ['Flooding', 'Fuzzing', 'Normal', 'Replay', 'Spoofing']
ml_rows = {cls: {
    'Precision': ml_metrics[cls]['precision'],
    'Recall': ml_metrics[cls]['recall'],
    'F1-score': ml_metrics[cls]['f1-score']
} for cls in classes}

df_ml = pd.DataFrame.from_dict(ml_rows, orient='index')

# Plot grouped bar chart for ML_classifier
ax2 = df_ml.plot(kind='bar', figsize=(8, 5))
ax2.set_title("RandomForest Model Performance by Class")
ax2.set_ylabel("Score")
ax2.set_ylim(0.7, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()

# export to png
plt.savefig('metrics\\ML_classifier_performance.png', dpi=300)

plt.show()
