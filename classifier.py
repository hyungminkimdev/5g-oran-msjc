# import uhd
# import numpy as np
# import torch
# from scipy import signal
# import time

# # ... 그 아래에 기존 코드들 (class O_RAN_JammingCNN 등) ...

# def iq_to_tensor(iq_chunk, nfft=128):
#     # Calculate Spectrogram
#     f, t, Sxx = signal.spectrogram(iq_chunk, fs=20e6, nperseg=nfft, noverlap=nfft//2)
#     # Normalize and convert to decibels
#     Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
#     # Reshape for CNN: (Channels, Height, Width) -> (1, 128, M)
#     tensor = torch.from_numpy(Sxx_db).float().unsqueeze(0)
#     return tensor

# def calculate_oran_kpis(iq_chunk):
#     power = np.mean(np.abs(iq_chunk)**2)
#     rssi = 10 * np.log10(power + 1e-12)
#     # Spectral Flatness: Geometric mean / Arithmetic mean of PSD
#     psd = np.abs(np.fft.fft(iq_chunk))**2
#     flatness = np.exp(np.nanmean(np.log(psd + 1e-12))) / np.nanmean(psd)
#     return rssi, flatness

# class O_RAN_JammingCNN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.AdaptiveAvgPool2d((8, 8))
#         )
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(64 * 8 * 8, 256),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.4),
#             torch.nn.Linear(256, 4) # 4 Classes: None, Constant, Random, Reactive
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)

# def production_inference_loop(usrp_addr, model_path=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = O_RAN_JammingCNN().to(device)
#     if model_path: model.load_state_dict(torch.load(model_path))
#     model.eval()

#     usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")

#     # === 여기서 주파수와 대역폭을 수정합니다 ===
#     rate = 20e6               # 대역폭 (송신기와 동일하게)
#     freq = 3.5e9              # 주파수 (송신기와 동일하게: 3.5GHz)

#     usrp.set_rx_gain(45) # 수신 감도 상향
#     usrp.set_rx_antenna("RX2", 0) # 보통 X300은 RX2 포트를 수신용으로 씀
#     streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    
#     # Large buffer for m1.8xlarge
#     buffer = np.zeros((1, 1024*512), dtype=np.complex64) 
#     metadata = uhd.types.RXMetadata()

#     stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
#     stream_cmd.stream_now = True
#     streamer.issue_stream_cmd(stream_cmd)

#     try:
#         while True:
#             num_recv = streamer.recv(buffer, metadata)
#             if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
#                 print(f"UHD Error: {metadata.strerror()}")
#                 continue
            
#             # 1. KPI Extraction
#             rssi, flatness = calculate_oran_kpis(buffer[0])
            
#             # 2. Pre-process for CNN
#             input_tensor = iq_to_tensor(buffer[0][:128*128]).to(device).unsqueeze(0)
            
#             # 3. Inference
#             with torch.no_grad():
#                 outputs = model(input_tensor)
#                 _, predicted = torch.max(outputs, 1)
                
#             classes = ['No Jamming', 'Constant', 'Random', 'Reactive']
#             print(f"[KPI] RSSI: {rssi:.2f} | Flatness: {flatness:.4f} | DETECTED: {classes[predicted]}")

#     except KeyboardInterrupt:
#         stream_cmd.stream_mode = uhd.types.StreamMode.stop_cont
#         streamer.issue_stream_cmd(stream_cmd)

# # ... (앞부분 클래스 및 함수 정의 생략) ...

# if __name__ == "__main__":
#     # 실시간 분석 루프 실행
#     production_inference_loop(usrp_addr="192.168.115.2")


import uhd
import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt # MODIFIED: 시각화 라이브러리 추가
import time
import os

def iq_to_tensor(iq_chunk, nfft=128):
    f, t, Sxx = signal.spectrogram(iq_chunk, fs=20e6, nperseg=nfft, noverlap=nfft//2)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    # 시각화 데이터 반환을 위해 Sxx_db도 같이 리턴하도록 변경 가능하지만, 
    # 여기서는 텐서만 리턴하고 시각화는 루프 안에서 처리하겠습니다.
    tensor = torch.from_numpy(Sxx_db).float().unsqueeze(0)
    return tensor, Sxx_db # MODIFIED: 시각화용 DB값도 같이 반환


def calculate_oran_kpis(iq_chunk):
    power = np.mean(np.abs(iq_chunk)**2)
    rssi = 10 * np.log10(power + 1e-12)
    # Spectral Flatness: Geometric mean / Arithmetic mean of PSD
    psd = np.abs(np.fft.fft(iq_chunk))**2
    flatness = np.exp(np.nanmean(np.log(psd + 1e-12))) / np.nanmean(psd)
    return rssi, flatness

class O_RAN_JammingCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 4) # 4 Classes: None, Constant, Random, Reactive
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def production_inference_loop(usrp_addr, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = O_RAN_JammingCNN().to(device)
    
    # 모델 파일이 실제로 있을 때만 로드 (없으면 기본 가중치 사용)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()

    usrp = uhd.usrp.MultiUSRP(f"addr={usrp_addr}")
    rate = 20e6
    freq = 3.5e9

    usrp.set_rx_rate(rate) # 추가: 수신 속도 명시적 설정
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq)) # 추가: 주파수 설정 확인
    usrp.set_rx_gain(45) 
    usrp.set_rx_antenna("RX2", 0) 
    
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    buffer = np.zeros((1, 1024*512), dtype=np.complex64) 
    metadata = uhd.types.RXMetadata()

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    print("--- 실시간 모니터링 시작 ---")
    save_idx = 0

    try:
        while True:
            num_recv = streamer.recv(buffer, metadata)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            
            # 1. KPI Extraction
            rssi, flatness = calculate_oran_kpis(buffer[0])
            
            # 2. Pre-process for CNN
            input_tensor, spec_data = iq_to_tensor(buffer[0][:128*128])
            input_tensor = input_tensor.to(device).unsqueeze(0)
            
            # 3. Inference & Logic 수정
            with torch.no_grad():
                # --- MODIFIED: 로직 강화 (RSSI 기반 필터링) ---
                # 신호 세기가 너무 낮으면(-60dB 이하 등) AI 판단과 상관없이 'No Jamming'으로 간주
                if rssi < -50:
                    predicted = 0 # 'No Jamming' 클래스 인덱스
                else:
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                
            classes = ['No Jamming', 'Constant', 'Random', 'Reactive']
            print(f"[KPI] RSSI: {rssi:.2f} | Flatness: {flatness:.4f} | DETECTED: {classes[predicted]}")

            # 4. MODIFIED: 데이터 가시화 (30번 루프마다 한 번씩 스펙트로그램 이미지 저장)
            if save_idx % 30 == 0:
                plt.figure(figsize=(10, 4))
                plt.imshow(spec_data, aspect='auto', origin='lower')
                plt.colorbar(label='dB')
                plt.title(f"Detected: {classes[predicted]} (RSSI: {rssi:.2f})")
                plt.savefig(f"spectrogram_{save_idx}.png")
                plt.close()
                print(f" >>> 시각화 이미지 저장됨: spectrogram_{save_idx}.png")
            
            save_idx += 1

    except KeyboardInterrupt:
        stream_cmd.stream_mode = uhd.types.StreamMode.stop_cont
        streamer.issue_stream_cmd(stream_cmd)

if __name__ == "__main__":
    # 실시간 분석 루프 실행
    production_inference_loop(usrp_addr="192.168.115.2")