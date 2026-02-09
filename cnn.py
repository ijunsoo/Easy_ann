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
    (64, 3), (64, 5),
    (128, 2), (128, 3), (128, 5),
]

# ==========================================
# 2. 장치 설정
# ==========================================
devices = [torch.device("cpu")]
if torch.backends.mps.is_available():
    devices.append(torch.device("mps"))
print(f"측정 대상 장치: {[str(d) for d in devices]}")

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
        in_channels = 1  # MNIST는 흑백 1채널

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** min(i, 2))  # 깊어질수록 필터 증가 (최대 4배)
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            # 2개 레이어마다 풀링 (공간 크기를 너무 빨리 줄이지 않기 위해)
            if (i + 1) % 2 == 0:
                conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers

        # FC 레이어 입력 크기 계산 (동적)
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
    print(f"\n{'='*60}")
    print(f"=== [{device.type.upper()}] CNN 실험 시작 ===")
    print(f"{'='*60}")

    # 데이터 사전 로딩 (전송 오버헤드 제거)
    print(f"  데이터를 {device.type.upper()}에 사전 로딩 중...")
    train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
    test_batches = [(d.to(device), t.to(device)) for d, t in test_loader]

    # MPS 워밍업
    if device.type == 'mps':
        dummy = torch.randn(1, 1, 28, 28, device=device)
        dummy_model = SimpleCNN(32, 2).to(device)
        _ = dummy_model(dummy)
        torch.mps.synchronize()
        del dummy, dummy_model
    print(f"  사전 로딩 완료.")

    device_results = []

    for num_filters, num_conv_layers in configs:
        # 파라미터 수 계산
        model_tmp = SimpleCNN(num_filters, num_conv_layers)
        param_count = sum(p.numel() for p in model_tmp.parameters())
        del model_tmp

        print(f"\n  --- 필터: {num_filters}, 레이어: {num_conv_layers} (파라미터: {param_count:,}) ---")
        train_times = []
        infer_times = []

        for i in range(REPEAT_COUNT):
            model = SimpleCNN(num_filters, num_conv_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

            # --- 학습 시간 측정 ---
            model.train()
            if device.type == 'mps': torch.mps.synchronize()
            start = time.perf_counter()

            for _ in range(EPOCHS_PER_RUN):
                for data, target in train_batches:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            if device.type == 'mps': torch.mps.synchronize()
            train_time = time.perf_counter() - start

            # --- 추론 시간 측정 ---
            model.eval()
            if device.type == 'mps': torch.mps.synchronize()
            start = time.perf_counter()

            correct = 0
            with torch.no_grad():
                for data, target in test_batches:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            if device.type == 'mps': torch.mps.synchronize()
            infer_time = time.perf_counter() - start
            accuracy = 100. * correct / len(test_dataset)

            train_times.append(train_time)
            infer_times.append(infer_time)
            print(f"    [{i+1}/{REPEAT_COUNT}] 학습: {train_time:.4f}s | 추론: {infer_time:.4f}s | 정확도: {accuracy:.2f}%")

        result = {
            'device': device.type,
            'model': 'CNN',
            'filters': num_filters,
            'conv_layers': num_conv_layers,
            'params': param_count,
            'avg_train': round(np.mean(train_times), 4),
            'std_train': round(np.std(train_times), 4),
            'avg_infer': round(np.mean(infer_times), 4),
            'std_infer': round(np.std(infer_times), 4),
        }
        device_results.append(result)
        print(f"    >> 평균 학습: {result['avg_train']}s | 평균 추론: {result['avg_infer']}s")

    all_results[device.type] = device_results

# ==========================================
# 6. 결과 출력
# ==========================================
print("\n\n" + "="*60)
print("   CNN 최종 결과 보고서")
print("="*60)

for dev_name, results in all_results.items():
    print(f"\n[{dev_name.upper()}]")
    print(f"  {'필터':>6} | {'레이어':>6} | {'파라미터':>10} | {'평균학습(s)':>10} | {'평균추론(s)':>10}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r in results:
        print(f"  {r['filters']:>6} | {r['conv_layers']:>6} | {r['params']:>10,} | {r['avg_train']:>10.4f} | {r['avg_infer']:>10.4f}")

# CPU vs MPS 비교
if len(all_results) == 2 and 'cpu' in all_results and 'mps' in all_results:
    print(f"\n[CPU vs MPS 비교]")
    print(f"  {'필터':>6} | {'레이어':>6} | {'CPU학습':>8} | {'MPS학습':>8} | {'배율':>6} | {'CPU추론':>8} | {'MPS추론':>8} | {'배율':>6}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6}")
    for cpu_r, mps_r in zip(all_results['cpu'], all_results['mps']):
        train_ratio = cpu_r['avg_train'] / mps_r['avg_train'] if mps_r['avg_train'] > 0 else 0
        infer_ratio = cpu_r['avg_infer'] / mps_r['avg_infer'] if mps_r['avg_infer'] > 0 else 0
        faster_train = "MPS" if train_ratio > 1 else "CPU"
        faster_infer = "MPS" if infer_ratio > 1 else "CPU"
        print(f"  {cpu_r['filters']:>6} | {cpu_r['conv_layers']:>6} | {cpu_r['avg_train']:>8.4f} | {mps_r['avg_train']:>8.4f} | {train_ratio:>5.2f}x | {cpu_r['avg_infer']:>8.4f} | {mps_r['avg_infer']:>8.4f} | {infer_ratio:>5.2f}x")

print("\n\n=== JSON 결과 ===")
print(json.dumps(all_results, indent=2))
