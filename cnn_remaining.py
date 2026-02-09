import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import json

# ==========================================
# 설정
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS_PER_RUN = 1
REPEAT_COUNT = 10

# 전체 설정 (MPS용)
all_configs = [
    (32, 2), (32, 3), (32, 5),
    (64, 3), (64, 5),
    (128, 2), (128, 3), (128, 5),
]

# CPU는 (128,5)만 남음
cpu_configs = [(128, 5)]

# ==========================================
# 데이터셋 준비
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("데이터 로딩 중...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
print("데이터 로딩 완료.\n")

# ==========================================
# 모델 정의
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
# 실험 함수
# ==========================================
def run_configs(device, configs, train_batches, test_batches):
    results = []
    for num_filters, num_conv_layers in configs:
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

            # 학습 시간 측정
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

            # 추론 시간 측정
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
        results.append(result)
        print(f"    >> 평균 학습: {result['avg_train']}s | 평균 추론: {result['avg_infer']}s")
    return results

# ==========================================
# 1단계: CPU (128,5)만 실행
# ==========================================
all_results = {}

cpu = torch.device("cpu")
print(f"{'='*60}")
print(f"=== [CPU] (128,5) 나머지 실험 ===")
print(f"{'='*60}")
print("  데이터를 CPU에 사전 로딩 중...")
train_batches_cpu = [(d.to(cpu), t.to(cpu)) for d, t in train_loader]
test_batches_cpu = [(d.to(cpu), t.to(cpu)) for d, t in test_loader]
print("  사전 로딩 완료.")

cpu_results = run_configs(cpu, cpu_configs, train_batches_cpu, test_batches_cpu)
all_results['cpu_128_5'] = cpu_results

# ==========================================
# 2단계: MPS 전체 8설정
# ==========================================
if torch.backends.mps.is_available():
    mps = torch.device("mps")
    print(f"\n\n{'='*60}")
    print(f"=== [MPS GPU] 전체 8설정 실험 ===")
    print(f"{'='*60}")
    print("  데이터를 MPS에 사전 로딩 중...")
    train_batches_mps = [(d.to(mps), t.to(mps)) for d, t in train_loader]
    test_batches_mps = [(d.to(mps), t.to(mps)) for d, t in test_loader]

    # MPS 워밍업
    dummy = torch.randn(1, 1, 28, 28, device=mps)
    dummy_model = SimpleCNN(32, 2).to(mps)
    _ = dummy_model(dummy)
    torch.mps.synchronize()
    del dummy, dummy_model
    print("  사전 로딩 + 워밍업 완료.")

    mps_results = run_configs(mps, all_configs, train_batches_mps, test_batches_mps)
    all_results['mps'] = mps_results
else:
    print("\nMPS를 사용할 수 없습니다.")

# ==========================================
# 결과 출력
# ==========================================
print("\n\n" + "="*60)
print("   CNN 잔여 실험 최종 결과")
print("="*60)

for key, results in all_results.items():
    print(f"\n[{key.upper()}]")
    print(f"  {'필터':>6} | {'레이어':>6} | {'파라미터':>10} | {'평균학습(s)':>10} | {'평균추론(s)':>10}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r in results:
        print(f"  {r['filters']:>6} | {r['conv_layers']:>6} | {r['params']:>10,} | {r['avg_train']:>10.4f} | {r['avg_infer']:>10.4f}")

print("\n\n=== JSON 결과 ===")
print(json.dumps(all_results, indent=2))
