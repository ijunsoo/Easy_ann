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

# 장치 설정 (GPU가 있으면 GPU, 없으면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"현재 사용 중인 하드웨어: {device}")

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
def run_experiment(iteration_idx):
    # 모델 초기화 (매 실험마다 가중치를 초기화하기 위해 새로 생성)
    model = SimpleANN(input_size=784, 
                      hidden_size=HIDDEN_SIZE, 
                      num_layers=NUM_HIDDEN_LAYERS, 
                      output_size=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # --- [학습 시간 측정] ---
    model.train()
    
    # GPU 시간 측정의 정확도를 위해 동기화 (CPU일 때는 영향 없음)
    if device.type == 'cuda': torch.cuda.synchronize()
    start_train = time.perf_counter()
    
    for epoch in range(EPOCHS_PER_RUN):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    if device.type == 'cuda': torch.cuda.synchronize()
    end_infer = time.perf_counter()
    infer_time = end_infer - start_infer
    
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"[{iteration_idx+1}/{REPEAT_COUNT}회차] 학습 시간: {train_time:.4f}초 | 추론 시간: {infer_time:.4f}초 | 정확도: {accuracy:.2f}%")
    
    return train_time, infer_time

# ==========================================
# 5. 메인 실행 루프
# ==========================================
print(f"=== 실험 시작 (Hidden: {HIDDEN_SIZE}, Layers: {NUM_HIDDEN_LAYERS}) ===")

train_times = []
infer_times = []

for i in range(REPEAT_COUNT):
    t_train, t_infer = run_experiment(i)
    train_times.append(t_train)
    infer_times.append(t_infer)

# ==========================================
# 6. 결과 통계 출력
# ==========================================
avg_train = np.mean(train_times)
std_train = np.std(train_times)
avg_infer = np.mean(infer_times)
std_infer = np.std(infer_times)

print("\n" + "="*40)
print(f"   최종 결과 보고서 ({REPEAT_COUNT}회 반복)")
print("="*40)
print(f"설정 변인 -> 뉴런 수: {HIDDEN_SIZE}, 은닉층 수: {NUM_HIDDEN_LAYERS}")
print("-" * 40)
print(f"평균 학습 시간 : {avg_train:.5f} sec (표준편차: {std_train:.5f})")
print(f"평균 추론 시간 : {avg_infer:.5f} sec (표준편차: {std_infer:.5f})")
print("="*40)