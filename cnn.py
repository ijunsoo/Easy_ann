import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import json

# ==========================================
# 1. 설정 (Hyperparameters & Settings)
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS_PER_RUN = 1
REPEAT_COUNT = 10

# (필터 수, 컨볼루션 레이어 수) - ANN의 (뉴런 수, 레이어 수)와 대응
configs = [
    (32, 2), (32, 3), (32, 5),
    (64, 2), (64, 3), (64, 5),
    (128, 2), (128, 3), (128, 5),
]

# ==========================================
# 2. 장치 설정 (Windows CUDA / Mac MPS 자동 감지)
# ==========================================
devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
elif torch.backends.mps.is_available():
    devices.append(torch.device("mps"))
print(f"측정 대상 장치: {[str(d) for d in devices]}")

def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def clear_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def gpu_label(device):
    if device.type == 'cuda':
        return "GPU(CUDA)"
    elif device.type == 'mps':
        return "GPU(MPS)"
    return "CPU"

# ==========================================
# 3. 데이터셋 준비 (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("데이터 다운로드 및 로딩 중...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
print("데이터 로딩 완료.\n")

# ==========================================
# 4. 모델 정의 (Flexible CNN)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_filters, num_conv_layers, num_classes=10):
        super(SimpleCNN, self).__init__()

        conv_layers = []
        in_channels = 1

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** min(i, 2))
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            if (i + 1) % 2 == 0:
                conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            dummy_out = self.features(dummy)
            self.flat_size = dummy_out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# 5. 실험 실행
# ==========================================
all_results = {}

for device in devices:
    dev_key = gpu_label(device)

    print(f"\n{'='*60}")
    print(f"=== [{dev_key}] CNN 실험 시작 ===")
    print(f"{'='*60}")

    # 데이터 사전 로딩 (전송 오버헤드 제거)
    print(f"  데이터를 {dev_key}에 사전 로딩 중...")
    train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
    test_batches = [(d.to(device), t.to(device)) for d, t in test_loader]

    # GPU 워밍업
    if device.type in ('cuda', 'mps'):
        dummy = torch.randn(1, 1, 28, 28, device=device)
        dummy_model = SimpleCNN(32, 2).to(device)
        _ = dummy_model(dummy)
        sync_device(device)
        del dummy, dummy_model
    print(f"  사전 로딩 완료.\n")

    device_results = []

    for num_filters, num_conv_layers in configs:
        model_tmp = SimpleCNN(num_filters, num_conv_layers)
        param_count = sum(p.numel() for p in model_tmp.parameters())
        del model_tmp

        print(f"  --- 필터: {num_filters}, 레이어: {num_conv_layers} (파라미터: {param_count:,}) ---")
        train_times = []
        infer_times = []
        accuracies = []

        for i in range(REPEAT_COUNT):
            model = SimpleCNN(num_filters, num_conv_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

            # --- 학습 시간 측정 ---
            model.train()
            sync_device(device)
            start = time.perf_counter()

            for _ in range(EPOCHS_PER_RUN):
                for data, target in train_batches:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            sync_device(device)
            train_time = time.perf_counter() - start

            # --- 추론 시간 측정 ---
            model.eval()
            sync_device(device)
            start = time.perf_counter()

            correct = 0
            with torch.no_grad():
                for data, target in test_batches:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            sync_device(device)
            infer_time = time.perf_counter() - start
            accuracy = 100. * correct / len(test_dataset)

            train_times.append(train_time)
            infer_times.append(infer_time)
            accuracies.append(accuracy)
            print(f"    [{i+1}/{REPEAT_COUNT}] 학습: {train_time:.4f}s | 추론: {infer_time:.4f}s | 정확도: {accuracy:.2f}%")

            del model
            clear_cache(device)

        result = {
            'filters': num_filters,
            'layers': num_conv_layers,
            'params': param_count,
            'avg_train': round(float(np.mean(train_times)), 5),
            'std_train': round(float(np.std(train_times)), 5),
            'avg_infer': round(float(np.mean(infer_times)), 5),
            'std_infer': round(float(np.std(infer_times)), 5),
            'avg_accuracy': round(float(np.mean(accuracies)), 2),
        }
        device_results.append(result)
        print(f"    >> 평균 학습: {result['avg_train']}s | 평균 추론: {result['avg_infer']}s | 정확도: {result['avg_accuracy']}%\n")

    all_results[dev_key] = device_results

    del train_batches, test_batches
    clear_cache(device)

# ==========================================
# 6. 결과 JSON 저장
# ==========================================
with open('cnn_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\n결과가 cnn_results.json에 저장되었습니다.")

# ==========================================
# 7. 결과 출력
# ==========================================
print("\n\n" + "="*60)
print(f"   CNN 최종 결과 보고서 ({REPEAT_COUNT}회 반복)")
print("="*60)

for dev_name, results in all_results.items():
    print(f"\n[{dev_name}]")
    print(f"  {'필터':>6} | {'레이어':>6} | {'파라미터':>10} | {'평균학습(s)':>11} | {'평균추론(s)':>11} | {'정확도':>7}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*11} | {'-'*11} | {'-'*7}")
    for r in results:
        print(f"  {r['filters']:>6} | {r['layers']:>6} | {r['params']:>10,} | {r['avg_train']:>11.5f} | {r['avg_infer']:>11.5f} | {r['avg_accuracy']:>6.2f}%")

# CPU vs GPU 비교
dev_keys = list(all_results.keys())
if len(dev_keys) == 2:
    cpu_key = dev_keys[0]
    gpu_key = dev_keys[1]
    print(f"\n[{cpu_key} vs {gpu_key} 비교]")
    print(f"  {'필터':>6} | {'레이어':>6} | {'CPU학습':>9} | {'GPU학습':>9} | {'학습배율':>8} | {'CPU추론':>9} | {'GPU추론':>9} | {'추론배율':>8}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*8}")
    for cpu_r, gpu_r in zip(all_results[cpu_key], all_results[gpu_key]):
        train_ratio = cpu_r['avg_train'] / gpu_r['avg_train'] if gpu_r['avg_train'] > 0 else 0
        infer_ratio = cpu_r['avg_infer'] / gpu_r['avg_infer'] if gpu_r['avg_infer'] > 0 else 0
        print(f"  {cpu_r['filters']:>6} | {cpu_r['layers']:>6} | {cpu_r['avg_train']:>9.5f} | {gpu_r['avg_train']:>9.5f} | {train_ratio:>7.2f}x | {cpu_r['avg_infer']:>9.5f} | {gpu_r['avg_infer']:>9.5f} | {infer_ratio:>7.2f}x")

print("\n" + "="*60)
print("CNN 벤치마크 완료!")
