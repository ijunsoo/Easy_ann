import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# ==========================================
# 1. 설정 (Hyperparameters & Settings)
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS_PER_RUN = 1        # 측정 1회당 학습할 에폭 수 (시간 절약을 위해 1로 설정)
REPEAT_COUNT = 10         # 반복 측정 횟수 (과제 요구사항: 10번 이상)

# 변인 통제 (이 값을 바꾸어가며 실험하세요)
HIDDEN_SIZE = 256        # 은닉층의 뉴런 개수
NUM_HIDDEN_LAYERS = 5     # 은닉층의 개수 (입력층, 출력층 제외)

# 측정할 장치 목록 (GPU가 있으면 둘 다, 없으면 CPU만)
devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
print(f"측정 대상 장치: {[str(d) for d in devices]}")

# ==========================================
# 2. 데이터셋 준비 (MNIST)
# ==========================================
# 데이터 전처리: 텐서 변환 및 정규화
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
# 3. 모델 정의 (Flexible ANN)
# ==========================================
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleANN, self).__init__()
        self.layers = nn.ModuleList()

        # 입력층 -> 첫 번째 은닉층
        self.layers.append(nn.Linear(input_size, hidden_size))

        # 추가 은닉층들
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # 마지막 은닉층 -> 출력층
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 이미지 평탄화 (Flatten): [Batch, 1, 28, 28] -> [Batch, 784]
        x = x.view(-1, 28 * 28)

        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)
        return x

# ==========================================
# 4. 시간 측정 함수 정의
# ==========================================
def run_experiment(device, iteration_idx, train_batches, test_batches):
    # 모델 초기화 (매 실험마다 가중치를 초기화하기 위해 새로 생성)
    model = SimpleANN(input_size=784,
                      hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_HIDDEN_LAYERS,
                      output_size=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # --- [학습 시간 측정] (데이터는 이미 device에 올라가 있으므로 순수 연산만 측정) ---
    model.train()

    if device.type == 'cuda': torch.cuda.synchronize()
    start_train = time.perf_counter()

    for _ in range(EPOCHS_PER_RUN):
        for data, target in train_batches:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    if device.type == 'cuda': torch.cuda.synchronize()
    end_train = time.perf_counter()
    train_time = end_train - start_train

    # --- [추론(테스트) 시간 측정] ---
    model.eval()

    if device.type == 'cuda': torch.cuda.synchronize()
    start_infer = time.perf_counter()

    correct = 0
    with torch.no_grad():
        for data, target in test_batches:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if device.type == 'cuda': torch.cuda.synchronize()
    end_infer = time.perf_counter()
    infer_time = end_infer - start_infer

    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"  [{iteration_idx+1}/{REPEAT_COUNT}회차] 학습: {train_time:.4f}초 | 추론: {infer_time:.4f}초 | 정확도: {accuracy:.2f}%")

    return train_time, infer_time

# ==========================================
# 5. 메인 실행 루프 (장치별 측정)
# ==========================================
all_results = {}

for device in devices:
    print(f"\n{'='*50}")
    print(f"=== [{device.type.upper()}] 실험 시작 (Hidden: {HIDDEN_SIZE}, Layers: {NUM_HIDDEN_LAYERS}) ===")
    print(f"{'='*50}")

    # 측정 전에 전체 데이터를 device에 미리 올려둠 (전송 오버헤드 제거)
    print(f"  데이터를 {device.type.upper()}에 사전 로딩 중...")
    train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
    test_batches = [(d.to(device), t.to(device)) for d, t in test_loader]
    print(f"  사전 로딩 완료. 순수 연산 시간만 측정합니다.")

    train_times = []
    infer_times = []

    for i in range(REPEAT_COUNT):
        t_train, t_infer = run_experiment(device, i, train_batches, test_batches)
        train_times.append(t_train)
        infer_times.append(t_infer)

    all_results[device.type] = {
        'train_times': train_times,
        'infer_times': infer_times,
    }

# ==========================================
# 6. 결과 통계 출력
# ==========================================
print("\n" + "="*60)
print(f"   최종 결과 보고서 ({REPEAT_COUNT}회 반복)")
print(f"   설정 변인 -> 뉴런 수: {HIDDEN_SIZE}, 은닉층 수: {NUM_HIDDEN_LAYERS}")
print("="*60)

for dev_name, result in all_results.items():
    avg_train = np.mean(result['train_times'])
    std_train = np.std(result['train_times'])
    avg_infer = np.mean(result['infer_times'])
    std_infer = np.std(result['infer_times'])

    print(f"\n[{dev_name.upper()}]")
    print(f"  평균 학습 시간 : {avg_train:.5f} sec (표준편차: {std_train:.5f})")
    print(f"  평균 추론 시간 : {avg_infer:.5f} sec (표준편차: {std_infer:.5f})")

if len(all_results) == 2:
    cpu_train = np.mean(all_results['cpu']['train_times'])
    gpu_train = np.mean(all_results['cuda']['train_times'])
    cpu_infer = np.mean(all_results['cpu']['infer_times'])
    gpu_infer = np.mean(all_results['cuda']['infer_times'])

    print(f"\n[비교]")
    print(f"  학습 속도 비율 : CPU/GPU = {cpu_train/gpu_train:.2f}x")
    print(f"  추론 속도 비율 : CPU/GPU = {cpu_infer/gpu_infer:.2f}x")

print("="*60)
