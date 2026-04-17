import uhd
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
import time
import os
from sklearn.mixture import GaussianMixture

# 1. CNN 모델 정의 (기존과 동일)
class O_RAN_JammingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2) # SAJD 핵심: Clean(0) vs Interference(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. 유틸리티 함수들
def iq_to_spectrogram(iq_chunk, nfft=128):
    f, t, Sxx = signal.spectrogram(iq_chunk, fs=20e6, nperseg=nfft, noverlap=nfft//2)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    return Sxx_db

def calculate_kpis(iq_chunk):
    power = np.mean(np.abs(iq_chunk)**2)
    rssi = 10 * np.log10(power + 1e-12)
    psd = np.abs(np.fft.fft(iq_chunk[:1024]))**2
    flatness = np.exp(np.nanmean(np.log(psd + 1e-12))) / np.nanmean(psd)
    return rssi, flatness

# # 3. [모드 1] 자동 라벨러 (Labeler rApp 로직)
# def run_auto_labeler(usrp_addr, duration=30):
    print(f"--- [단계 1] 데이터 수집 및 자동 라벨링 시작 ({duration}초) ---")
    usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")
    usrp.set_rx_gain(45)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    
    buffer = np.zeros((1, 128*128), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    features, specs = [], []
    
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    start_t = time.time()
    while time.time() - start_t < duration:
        num_recv = streamer.recv(buffer, metadata)
        if num_recv > 0:
            rssi, flatness = calculate_kpis(buffer[0])
            features.append([rssi, flatness])
            specs.append(iq_to_spectrogram(buffer[0]))

    # GMM 클러스터링 (비지도 학습 라벨링)
    X = np.array(features)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    labels = gmm.predict(X)
    
    # RSSI가 낮은 쪽을 Clean(0)으로 설정
    if gmm.means_[0][0] > gmm.means_[1][0]:
        labels = 1 - labels

    dataset = {'x': torch.tensor(np.array(specs)).float().unsqueeze(1), 'y': torch.tensor(labels).long()}
    torch.save(dataset, "captured_dataset.pth")
    print(f"✅ 완료: {len(labels)}개 데이터 저장됨 -> captured_dataset.pth")

# 3. [수정된 모드 1] 더 가벼운 자동 라벨러
def run_auto_labeler(usrp_addr, duration=60): # 시간을 20초로 단축
    print(f"--- [단계 1] Raw 데이터 고속 수집 시작 ({duration}초) ---")
    usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")
    usrp.set_rx_rate(1e6) # 1MHz로 고정
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(3.5e9))
    usrp.set_rx_gain(45)
    
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    buffer = np.zeros((1, 128*128), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    raw_iq_samples = [] # IQ 샘플 자체를 저장
    features = []
    
    streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    start_t = time.time()
    while time.time() - start_t < duration:
        num_recv = streamer.recv(buffer, metadata)
        if num_recv > 0:
            # 1. 가벼운 KPI만 실시간 추출 (RSSI, Flatness)
            rssi, flatness = calculate_kpis(buffer[0])
            features.append([rssi, flatness])
            
            # 2. IQ 데이터는 복사본만 저장 (연산 안 함)
            raw_iq_samples.append(buffer[0].copy())

    print("--- [데이터 수집 완료] 이제 스펙트로그램 및 라벨 생성을 시작합니다 (Post-processing) ---")
    
    # 수집이 끝난 뒤에 무거운 연산 시작
    specs = [iq_to_spectrogram(iq) for iq in raw_iq_samples]
    
    # GMM 클러스터링
    X = np.array(features)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    labels = gmm.predict(X)
    
    if gmm.means_[0][0] > gmm.means_[1][0]:
        labels = 1 - labels

    dataset = {
        'x': torch.tensor(np.array(specs)).float().unsqueeze(1),
        'y': torch.tensor(labels).long()
    }
    torch.save(dataset, "captured_dataset.pth")
    print(f"✅ 최종 완료: {len(labels)}개 샘플 저장됨.")

# # 4. [모드 2] 모델 학습 (Training Manager 로직)
# def train_model(dataset_path):
    print("--- [단계 2] 모델 학습 시작 ---")
    data = torch.load(dataset_path)
    X, y = data['x'], data['y']
    
    model = O_RAN_JammingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(outputs, 1)
        acc = (pred == y).float().mean()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

    torch.save(model.state_dict(), "jamming_model.pth")
    print("✅ 완료: 모델 저장됨 -> jamming_model.pth")

# 4. [수정된 모드 2] 모델 학습 (미니 배치 적용)
from torch.utils.data import DataLoader, TensorDataset

def train_model(dataset_path):
    print("--- [단계 2] 모델 학습 시작 (미니 배치 적용) ---")
    data = torch.load(dataset_path)
    X, y = data['x'], data['y']
    
    # 데이터를 64개씩 쪼개서 처리할 수 있도록 DataLoader 생성 (논문 설정 준수)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = O_RAN_JammingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(loader):.4f} | Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), "jamming_model.pth")
    print("✅ 완료: 모델 저장됨 -> jamming_model.pth")

# 5. [모드 3] 실시간 감지 및 드리프트 체크 (Detection xApp 로직)
def run_inference(usrp_addr):
    print("--- [단계 3] 실시간 지능형 감지 시작 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = O_RAN_JammingCNN().to(device)
    model.load_state_dict(torch.load("jamming_model.pth"))
    model.eval()

    usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")
    usrp.set_rx_gain(45)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    buffer = np.zeros((1, 128*128), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    while True:
        streamer.recv(buffer, metadata)
        spec = iq_to_spectrogram(buffer[0])
        input_tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        status = "JAMMING" if pred.item() == 1 else "CLEAN"
        print(f"[{status}] Confidence: {conf.item():.2f}")

if __name__ == "__main__":
    print("\n1: 데이터 수집/라벨링 | 2: 모델 학습 | 3: 실시간 탐지")
    choice = input("모드를 선택하세요: ")
    addr = "192.168.115.2"
    
    if choice == "1": run_auto_labeler(addr)
    elif choice == "2": train_model("captured_dataset.pth")
    elif choice == choice == "3": run_inference(addr)

