# ESP32 CSI 호흡 파형과 숨참기 구간 탐지

팀 ESP32에서 수집한 패킷 CSV 한 개를 호흡 대역 후보 파형으로 변환하는 오프라인 Python 프로젝트다. P0는 Alzaabi 등의 웨이블릿·PCA 절차를 사용한다. P1은 사람이 기록한 시작·종료 시각으로 자발적 숨참기 구간을 학습한다. 두 단계는 독립적으로 실행할 수 있다.

## 설치와 첫 실행

Python 3.11 이상을 사용한다. Windows PowerShell 예시이며 다른 운영체제에서는 해당 가상환경을 활성화한 뒤 같은 Python 명령을 사용한다.

```powershell
git clone https://github.com/mjsung1122/csi-breathhold.git
cd csi-breathhold
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
python -m csi_breathhold p0 "data/capture.csv" --profile team_esp32_ht20_v1 --recording-id session01 --out runs/session01
```

자체 데이터가 없어도 다음 명령으로 확인할 수 있다. 데모는 모의 ESP32 패킷부터 같은 P0를 통과하는 120초 합성 기록 6개를 만든다. 각 실행에는 새 출력 폴더 이름을 사용한다.

```powershell
python -m csi_breathhold demo --out runs/demo
python -m pip install ".[ai,test]"
python -m csi_breathhold train runs/demo/manifest.csv --epochs 30 --out runs/smoke-model
python -m csi_breathhold predict runs/smoke-model/model.pt runs/demo/synthetic_05/p0/waveform.csv --out runs/smoke-pred
python -m csi_breathhold evaluate runs/smoke-model/model.pt runs/demo/manifest.csv --out runs/smoke-eval
python -m pytest -q
```

P0와 데모 생성은 CPU에서 실행된다. PyTorch는 P1용 `ai` extra로 설치한다. 데모의 시간 라벨과 CSI는 합성이며, 합성 체크포인트의 출처는 학습·예측·평가 JSON에 기록된다.

## P0 입력과 출력

한 행은 수신 패킷 하나다. 현재 지원하는 프로파일은 팀의 ESP32 20 MHz, 256바이트 L-LTF + HT-LTF 구성이다. 원본 7열 `seq,lost,ts_us,rssi,first_word_invalid,len,csi`를 보존한다. `ts_us`는 장치 시각(µs), `csi`는 부호 있는 int8의 허수부·실수부 쌍이다. 기존 6열 CSV처럼 첫 워드 플래그가 없거나 유효하지 않은 첫 워드가 하나라도 있으면 L-LTF에 공통으로 남는 51개 슬롯을 사용한다. 첫 워드가 모두 정상이면 52개다. 기본값은 L-LTF이며 `--view htltf`는 별도 분석용이다. 두 필드를 함께 쌓지 않는다. 데이터 순서나 길이가 다른 장치는 명시적인 프로파일이 필요하다. [팀 수집기](https://github.com/clapppp/csi-sensing/tree/66716ed03dc7147e9c175805bbc01d5178ca5cc0)

1. 유효한 부반송파의 I/Q를 진폭으로 바꾸고 장치의 첫 패킷 시각을 0초로 삼는다. 누락된 패킷은 이후 시각을 당기지 않는다.
2. 공백이 0.25초를 초과하면 구간을 나눈다. 각 구간의 진폭을 시간 보간하고 Fourier 방식으로 40 Hz에 재표본화한다.
3. 각 부반송파에 db4 이산 웨이블릿 7단계 분해를 적용하고 D6·D7만 역변환한다. 명목 대역은 0.15625–0.625 Hz(약 9.4–37.5회/분)다.
4. 연속 구간별로 채널 평균을 제거한 PCA에서 두 번째 주성분(PC2)을 고정해 추출한다. BNR, 목표 BPM, 라벨, 기준 호흡 파형을 사용하지 않는다.

성공하면 `waveform.csv`(시각·PC2·유효 여부·구간 ID), `waveform.png`, `psd.csv`, `psd.png`, `capture.json`, `run.json`을 저장한다. 유효 표시에는 신호 처리에 충분한 표본이 있다는 의미가 있다. 빈방이나 손 움직임에서도 후보 곡선이 나올 수 있다. PC2의 크기는 상대값이며, 부호에 들숨·날숨 의미를 부여하지 않는다.

7단계 분해에는 연속 896표본(40 Hz에서 22.4초)이 필요하다. 각 구간 양끝 5초를 제외하고 유효한 2차 주성분이 없으면 파형 칸을 비운다. 이때 종료코드는 2이며 이유는 `run.json`에 남는다. 형식·파일 오류는 종료코드 1이다.

### 논문 재현 범위

원 논문은 ESP32에서 120패킷/초로 수집해 40 Hz로 맞추고, 52개 채널의 db4 D6·D7 복원과 PC2 스펙트럼 피크를 이용해 **호흡수**를 벨트와 비교했다. 이 구현은 그 추출 단계를 따른다. 팀 수집률은 약 50패킷/초이며, 공백 분리, 무효 슬롯 처리, 반사 경계, 양끝 5초 제외, 구간별 PCA와 Hann 주기도 계산은 구현 시 정한 조건이다. 경계 제외로 모든 필터 영향을 제거할 수는 없다. 파형 모양의 정확도와 숨참기 탐지 성능은 각각 별도로 평가해야 한다. [Alzaabi et al., IEEE Access (2024)](https://doi.org/10.1109/ACCESS.2024.3349700)

## P1 라벨과 평가

CSV의 시각은 P0와 같은 장치 기록의 시작을 기준으로 한 초다. 관찰한 자연 호흡과 자발적 숨참기에 대해 시작 시각 이상, 종료 시각 미만의 구간을 기록한다. 아래는 형식 예시다.

```csv
recording_id,start_s,end_s,label
S01,10,42.5,natural_breathing
S01,42.5,54.2,voluntary_breath_hold
S01,54.2,80,natural_breathing
```

학습 목록 예시는 다음과 같다. 파형과 라벨 파일 경로는 목록 파일을 기준으로 한다. 한 참가자나 한 원본 기록에서 잘라낸 여러 조각은 같은 분할에 둔다. 여러 조각 또는 두 LTF를 별도 행으로 전달할 때는 동일한 `recording_group_id`를 지정한다.

```csv
recording_id,participant_id,recording_group_id,split,waveform,labels,data_origin
S01,P01,S01,train,S01/waveform.csv,intervals.csv,real
S02,P02,S02,val,S02/waveform.csv,intervals.csv,real
S03,P03,S03,test,S03/waveform.csv,intervals.csv,real
```

P1은 각 연속 구간에서 PC2의 상대 진폭과 변화율을 구한다. 유효한 8초 창 안에서 1초 단위로 숨참기 점수를 계산한다. 시간 방향 합성곱 3개(채널 12, 팽창률 1·2·4)를 사용하며, 점수 0.5 이상이 이어지는 초를 하나의 후보 구간으로 저장한다. 이 임곗값은 실험 설정이다. 전체 구간으로 PCA와 정규화를 계산하고 창 중심의 앞뒤 데이터를 사용하므로 녹화가 끝난 파일에서 실행한다.

- **학습**은 `train`과 `val` 양쪽에 실제로 점수를 낼 수 있는 두 클래스의 시간 라벨이 필요하다. 기본값은 AdamW, 학습률 0.003, 배치 64, 30 epoch이며 검증 손실이 가장 낮은 가중치를 `model.pt`로 저장한다. 라벨 공백, 부분적으로만 라벨된 1초, CSI 공백·경계는 손실에서 제외한다.
- **예측**은 체크포인트와 `waveform.csv`만 받는다. `scores.csv`는 1초 점수, `events.csv`는 후보 구간, `prediction.json`은 판정 가능 시간과 모델 출처를 담는다. 판정 불가 시간의 점수는 빈칸이다.
- **평가**는 학습·검증과 분리된 `test`의 `evaluation.json`에 판정 가능 비율, 초 단위 혼동행렬, 사건 단위 precision·recall을 기록한다. 사건 평가는 라벨과 점수가 모두 있는 조각에서 시간 IoU 0.3 이상을 일대일로 매칭한다. 양성 구간 전체가 제외될 수 있으므로 양성 라벨 시간과 실제 평가 시간을 함께 확인한다.

팀 실측에는 현재 숨을 멈춘 양성 구간이 없다. 실측 모델 학습에는 여러 참가자의 동기화된 숨참기·자연 호흡 구간이 필요하다. 빈방은 별도 대조 조건이다. 이 프로젝트에서 검증할 대상은 **관찰자가 표시한 자발적 숨참기**다. 수면무호흡의 임상 판별에는 별도의 임상 기준 측정이 필요하다.

## 첫 실행 검증

팀 CSV 6개를 기본 L-LTF로 검사했다. 자연 호흡 3분 기록 3개에서는 각각 약 170초, 빈방과 반복 손 움직임 1분 기록에서는 각각 약 50초의 PC2가 유효했다. 10초 기록은 7단계 웨이블릿에 필요한 길이를 충족하지 못해 유효 표본이 0개였다. 팀 실측에는 동시 기준 호흡 파형이 없어 이 수치는 처리 가능한 시간을 나타낸다.

위 데모를 CPU에서 30 epoch 학습한 합성 시험에서는 2개 기록 240초 중 204초가 평가 가능했고 합성 숨참기 4구간 모두가 예측 구간과 대응했다. 초 단위 TP 61, FP 0, TN 142, FN 1이었다. 인위적으로 진폭을 줄인 합성 입력으로 학습·저장·예측·평가 과정을 검사한 결과다. 실제 참가자의 숨참기 탐지 성능은 아직 측정하지 않았다.
