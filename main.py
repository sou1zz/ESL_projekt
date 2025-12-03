import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time
import torch.nn.utils.prune as prune
from codecarbon import EmissionsTracker
import os
import json
import warnings

warnings.filterwarnings("ignore")

#config
BATCH_SIZE = 64
NUM_BATCHES = 15  
RESULTS_FILE = "test_results.json"

def get_data():
    print("Downloading CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)
    return loader

def benchmark_model(model, device, loader, description):
    print(f"\n>>> Testing: {description} on {device}...")
    model.to(device)
    
    if device == 'cuda':
        dummy = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        model(dummy)

    #code carbon
    tracker = EmissionsTracker(output_dir="./", log_level="error", save_to_file=False)
    tracker.start()
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= NUM_BATCHES: break
            inputs, _ = data
            inputs = inputs.to(device)
            _ = model(inputs)
            
    end_time = time.time()
    emissions = tracker.stop()
    
    total_time = end_time - start_time

    avg_latency = (total_time / (NUM_BATCHES * BATCH_SIZE)) * 1000 
    
    print(f"   Total time: {total_time:.2f} s")
    print(f"   Latency: {avg_latency:.2f} ms/image")
    print(f"   CO2 emission: {emissions:.8f} kg")
    
    return avg_latency, emissions

def main():
    results = {}
    loader = get_data()
    
    #base model
    print("\n Loading MobileNetV2 ---")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()

    #fp32 size
    torch.save(model.state_dict(), "temp_fp32.pt")
    size_fp32 = os.path.getsize("temp_fp32.pt") / 1e6

    #cpu test
    lat_cpu, em_cpu = benchmark_model(model, "cpu", loader, "CPU (Standard)")
    results['CPU (Standard)'] = {'latency': lat_cpu, 'energy': em_cpu}

    #gpu test
    if torch.cuda.is_available():
        lat_gpu, em_gpu = benchmark_model(model, "cuda", loader, "GPU (CUDA)")
        results['GPU (CUDA)'] = {'latency': lat_gpu, 'energy': em_gpu}
    else:
        print("\n[INFO] No NVIDIA GPU found. Skipping GPU test...")
        results['GPU (CUDA)'] = {'latency': 0, 'energy': 0} 

    #pruning
    print("\n Pruning")
    model_pruned = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for name, module in model_pruned.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
    
    lat_pruned, em_pruned = benchmark_model(model_pruned, "cpu", loader, "CPU (Pruned)")
    results['CPU (Pruned)'] = {'latency': lat_pruned, 'energy': em_pruned}

    #quantization
    print("\n Quantization (INT8) ---")
    model_to_quant = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model_to_quant.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model_to_quant, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    torch.save(quantized_model.state_dict(), "temp_int8.pt")
    size_int8 = os.path.getsize("temp_int8.pt") / 1e6
    
    lat_quant, em_quant = benchmark_model(quantized_model, "cpu", loader, "CPU (INT8)")
    results['CPU (INT8)'] = {'latency': lat_quant, 'energy': em_quant}

    #saving size data
    results['sizes'] = {'FP32': size_fp32, 'INT8': size_int8}

    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FILE}. To see the plots run: make_plots.py!")

    try:
        os.remove("temp_fp32.pt")
        os.remove("temp_int8.pt")
    except: pass

if __name__ == '__main__':
    main()