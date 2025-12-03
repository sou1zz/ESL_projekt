import matplotlib.pyplot as plt
import json
import os
import sys

RESULTS_FILE = "test_results.json"

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found {RESULTS_FILE}. Run main.py! first")
        sys.exit(1)

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    #date
    scenarios = []
    latencies = []
    
    for key in ['CPU (Standard)', 'GPU (CUDA)', 'CPU (Pruned)', 'CPU (INT8)']:
        if key in data and data[key]['latency'] > 0:
            scenarios.append(key)
            latencies.append(data[key]['latency'])
    
    sizes_labels = list(data['sizes'].keys())
    sizes_vals = list(data['sizes'].values())

    plt.style.use('ggplot')
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']

    # latency plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, latencies, color=colors[:len(scenarios)])
    plt.title('Latency - less = better', fontsize=14)
    plt.ylabel('Time [ms]', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig('latency_plot.png', dpi=300)
    print("generated latency_plot.png")

    #fps plot
    fps_vals = [1000/x for x in latencies]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, fps_vals, color=colors[:len(scenarios)])
    plt.title('FPS - more = better', fontsize=14)
    plt.ylabel('FPS', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig('fps_plot.png', dpi=300)
    print("Generated fps_plot.png")

    #size plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sizes_labels, sizes_vals, color=['#4c72b0', '#c44e52'], width=0.5)
    plt.title('Model size (compressed)', fontsize=14)
    plt.ylabel('MB', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig('size_plot.png', dpi=300)
    print("Generated size_plot.png")

if __name__ == '__main__':
    main()