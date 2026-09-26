# TEFFP (Target Exposure Factor Function Parameters) Seeker

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Kernels-Triton-6E4AFF?style=flat-square)
![CUDA](https://img.shields.io/badge/GPU-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

[![english-readme](https://img.shields.io/badge/Language-English-yellow.svg)](./README.md)

---



### 📖 프로젝트 소개 ###

**TEFFP Seeker** 는 **[ATM-Eta](https://github.com/kimlvis31/AutoTradeMachine_Eta)** 의 확장 애플리케이션으로 개발된 GPU 가속 백테스팅 및 파라미터 최적화 엔진입니다. ATM-Eta에서 추출한 분석 데이터를 입력받아, 사용자 정의 **TEF (Target Exposure Factor) 함수** 를 수만 개의 파라미터 세트에 대해 병렬로 시뮬레이션하고, 사용자가 선택한 평가 기준을 가장 잘 만족하는 파라미터 세트를 탐색합니다.

ATM-Eta에서도 CPU로 전략을 백테스트할 수 있지만, 좋은 파라미터를 찾으려면 값을 조금씩 바꿔가며 같은 시뮬레이션을 수없이 반복해야 합니다. TEFFP Seeker는 이 반복을 GPU로 옮깁니다. 각 파라미터 세트는 커스텀 **Triton** 커널의 독립된 레인으로 실행되며, 그 위에서 개체군 기반의 gradient 탐색이 파라미터 공간을 탐색합니다.

#### **핵심 기능**
* **대규모 병렬 백테스팅** — 각 파라미터 세트가 Triton 커널 안에서 독립적으로 시뮬레이션되므로, 분 단위 시장 데이터 전체 구간에 대한 수천 개의 백테스트가 단일 GPU에서 동시에 실행됩니다.
* **거래소 규칙을 반영한 거래 시뮬레이션** — 시뮬레이터는 ATM-Eta의 거래 규칙을 그대로 따릅니다. tick/step/quote 정밀도 반올림, 거래 수수료, isolated 및 cross 마진 계산, Full Stop Loss, 그리고 Binance의 계층형 유지증거금 테이블로 계산한 청산가가 포함됩니다.
* **개체군 기반 Gradient 탐색** — *seeker* 개체군이 수치 gradient와 Adam 방식의 업데이트로 파라미터 공간을 탐색하며, 주기적인 재배치로 성능이 낮은 seeker를 교체하여 지역 최적해에 갇히지 않도록 합니다.
* **설정 가능한 평가 기준** — 후보는 최종 잔고, 성장률, 변동성, 또는 샤프 지수와 유사한 복합 지표로 평가되며, 사용자가 정한 위험 한도를 넘는 파라미터 세트는 최대 낙폭 필터로 제외됩니다.
* **플러그인 방식의 TEF 함수** — 전략은 독립적인 `teff_*.py` 모듈로 작성되며, 실행 시 자동으로 인식됩니다. ATM-Eta와 동일한 TEF 인터페이스 계약(분석 데이터 입력, 목표 노출도 출력)이 적용됩니다.
* **ATM-Eta로 직접 내보내기** — 각 탐색의 최적 파라미터 세트는 바로 사용할 수 있는 ATM-Eta **Trade Configuration** 파일(`.tc`)로 내보내집니다.

---



### ▶️ 실행 방법 ###
애플리케이션을 실행하기 전에 **Python 3.11 이상** 과 최신 드라이버가 설치된 **NVIDIA GPU** 가 필요합니다. Windows에서는 Python 설치 시 `PATH` 에 추가하는 옵션을 선택해야 합니다. `requirements.txt` 에 명시된 모든 라이브러리는 setup 스크립트가 가상환경에 자동으로 설치합니다.

#### **Windows** 🪟
1. 루트 디렉토리에서 `setup.bat` 을 실행합니다. `.venv` 가 생성되고 필요한 라이브러리가 설치됩니다.
2. ATM-Eta에서 추출한 분석 데이터(`{name}_descriptor.json`, `{name}_data.npy`)를 `analysisData/` 폴더에 넣습니다.
3. `config.py` 에서 `MODE` 와 해당 모드의 설정을 지정합니다 ([실행 모드](#operating-modes) 참고).
4. 루트 디렉토리에서 `run.bat` 을 실행합니다. 애플리케이션이 시작됩니다.

#### **Linux** 🐧
1. 터미널에서 `chmod +x setup.sh run.sh` 명령을 실행합니다.
2. 루트 디렉토리에서 `setup.sh` 를 실행합니다. `.venv` 가 생성되고 필요한 라이브러리가 설치됩니다.
3. ATM-Eta에서 추출한 분석 데이터(`{name}_descriptor.json`, `{name}_data.npy`)를 `analysisData/` 폴더에 넣습니다.
4. `config.py` 에서 `MODE` 와 해당 모드의 설정을 지정합니다 ([실행 모드](#operating-modes) 참고).
5. 루트 디렉토리에서 `run.sh` 를 실행합니다. 애플리케이션이 시작됩니다.

> **참고:** `SEEK` 모드에서는 각 탐색을 시작하기 전에, 탐색에 사용할 배치 크기들에 대해 Triton autotuning warmup이 수행되며 몇 분 정도 걸릴 수 있습니다.

---



### ✅ 요구 사양 ###
* **운영체제**: Windows 10/11 또는 Linux
* **Python**: `3.11` 이상
* **GPU**: CUDA를 지원하는 NVIDIA GPU
* **RAM**: 4GB 이상
* **저장공간**: 4GB 이상

---



### 🧱 시스템 아키텍처 ###

TEFFP Seeker는 전략 개발 워크플로우에서 ATM-Eta의 다음 단계에 위치합니다. ATM-Eta는 시장 데이터 수집과 다중 시간대 분석을 담당하고, 분석 결과를 키마다 하나의 컬럼을 갖는 평탄한 데이터셋으로 추출합니다. TEFFP Seeker는 이 데이터셋을 입력받아 GPU에서 최적의 TEF 함수 파라미터를 탐색하고, 그 결과를 Trade Configuration 형태로 ATM-Eta에 다시 전달합니다.

<img src="./docs/teffpseeker_diagram_systemArchitecture.drawio.png" width="1000">

위 다이어그램은 한 번의 탐색 흐름을 나타내며, 세 단계로 구성됩니다.

* **데이터 준비** — ATM-Eta에서 추출한 Linearized Analysis에 대해, 선택한 TEF 함수가 필요로 하는 가격 및 분석 키가 있는지 확인하고, 첫 유효 종가 이전 구간을 잘라낸 뒤, 공백을 채우고 연속된 GPU 텐서로 변환합니다. 이 과정은 탐색 대상마다 한 번 수행됩니다.
* **탐색 루프** — `config.py` 의 seeker 파라미터에 따라, 거래 파라미터와 TEF 함수 모델이 정의한 파라미터 범위 안의 무작위 위치에 seeker들을 생성합니다. 매 단계마다 seeker들의 현재 위치로부터 테스트 파라미터 세트를 만들어 GPU에 배치 단위로 전달합니다. 모든 배치 처리가 끝나면 결과를 평가하고, Adam 방식의 옵티마이저와 자체 개발한 재배치 기법으로 seeker들을 이동시킵니다. 이 루프는 종료 조건을 만족할 때까지 반복됩니다.
* **결과 출력** — 최적 파라미터 세트와 개선 이력을 저장하고, 각 탐색 대상의 최적 세트를 ATM-Eta Trade Configuration으로 내보냅니다.

각 단계의 상세 내용은 [GPU 시뮬레이션 엔진](#gpu-simulation-engine)과 [파라미터 탐색](#parameter-search)에서 다룹니다.

#### **모듈별 역할**

| 모듈 | 역할 |
| :--- | :--- |
| `main.py` | 진입점. 선택한 모드(`TEST` / `SEEK` / `READ`)를 실행하고, 진행 상황 출력, 결과 저장, Trade Configuration 내보내기, 잔고 이력 시각화를 담당 |
| `config.py` | 사용자 설정: 수치 정밀도, 파라미터 테스트 대상, seeker 탐색 대상, 읽어올 결과, 실행 모드 |
| `exitFunction_base.py` | 핵심 엔진. 분석 데이터를 GPU 텐서로 전처리하고, seeker 상태 관리, 테스트 파라미터 세트 생성, 결과 평가, 시뮬레이션 배치 실행을 담당 |
| `exitFunction_models.py` | `teffunctions/` 폴더를 탐색하여 모든 `teff_*.py` 모듈을 사용 가능한 TEF 함수로 등록 |
| `teffunctions/simulatorFunctions.py` | 공용 Triton 구성 요소: 시뮬레이션 상태 초기화, 인터벌별 거래 처리, 청산가 및 유지증거금 계산, 잔고 추세 평가, autotune 설정 |
| `teffunctions/teff_*.py` | 사용자 정의 TEF 함수. 각 모듈은 파라미터 모델, 읽어올 분석 키, Triton 배치 커널을 정의 |

<br>

<a name="operating-modes"></a>
#### **실행 모드**

| 모드 | 설정 | 설명 |
| :--- | :--- | :--- |
| `TEST` | `PARAMETERTEST` | 완전히 지정된 파라미터 세트 하나를 시뮬레이션하고, 가격, 잔고, 추세선 대비 편차 이력을 그래프로 표시 |
| `SEEK` | `SEEKERTARGETS` | 목록의 모든 대상에 대해 파라미터 탐색을 실행하고, 결과 저장과 Trade Configuration 내보내기를 거쳐 새 결과를 읽어옴 |
| `READ` | `RCODETOREAD` | 저장된 결과를 불러와 현재 분석 데이터와 일치하는지 확인한 뒤, 기록된 상위 100개 파라미터 세트를 다시 시뮬레이션하여 시각적으로 비교 |

---



<a name="gpu-simulation-engine"></a>
### ⚡ GPU 시뮬레이션 엔진 ###

#### 🔹 **파라미터 세트당 하나의 레인**

시뮬레이션 배치는 2차원 문제입니다. 여러 파라미터 세트가 각각 같은 긴 시계열을 따라 진행합니다. TEFFP Seeker는 각 파라미터 세트를 Triton 프로그램의 레인 하나에 대응시키며, 각 레인은 자신의 잔고, 포지션, TEF 모델 상태를 레지스터에 유지한 채 전체 시계열을 순차적으로 진행합니다. 가격 및 분석 데이터는 모든 레인이 읽기 전용으로 공유하므로, 메모리 트래픽은 파라미터 세트 수가 아니라 데이터 길이에 비례합니다.

블록 크기, warp 수, 파이프라인 단계 수는 배치 크기를 키로 하여 19개 후보 설정 중에서 **Triton autotuning** 으로 선택됩니다. 전체 데이터로 autotuning을 하면 오래 걸리기 때문에, 탐색 전에 사용할 모든 배치 크기에 대해 1주일 분량의 데이터로 먼저 warmup을 수행합니다.

<br>

#### 🔹 **데이터 전처리**

시뮬레이션 전에, 추출된 분석 데이터는 연속된 GPU 텐서로 변환됩니다.

* 유효한 종가가 없는 앞쪽 행들을 잘라내고, 남은 비율을 **validity rate** 로 보고합니다.
* 누락된 OHLC 값은 마지막 유효 종가로 채우며, 채워진 셀의 비율을 **gap rate** 로 보고합니다.
* 분석 컬럼은 공백을 채우지 않고 그대로 불러옵니다. 누락된 신호를 어떻게 해석할지는 전략만이 알기 때문에, 누락된 분석 값의 처리는 각 TEF 함수에 맡깁니다.

<br>

#### 🔹 **거래 시뮬레이션**

단순화를 위해 모든 거래는 인터벌 종가에 체결되는 시장가 주문으로 가정합니다. 거래 수수료율은 고정되어 있지 않으며, 시뮬레이션할 계정의 수수료 등급과 주문 유형에 맞게 `config.py` 의 `tradingFee` 로 탐색 대상마다 설정할 수 있습니다.

매 인터벌마다 각 레인은 분석 데이터로부터 TEF 방향과 값을 계산한 뒤, `simulatorFunctions.py` 의 공용 거래 단계를 실행합니다.

1. **청산 조건 확인** — Full Stop Loss(immediate, close 기준)와 강제 청산을 해당 인터벌의 가장 불리한 가격에 대해 평가합니다. 청산가는 Binance의 계층형 유지증거금 테이블로 계산합니다. 같은 인터벌에서 여러 청산 조건이 발생하면, 시가에 더 가까운 쪽이 먼저 체결된 것으로 가정합니다.
2. **포지션 축소** — 강제 청산, 방향 전환, 또는 TEF 값이 0인 경우 포지션을 전량 청산합니다. 그 외에는 투입된 잔고가 목표(`Allocated Balance × |TEF|`)를 초과하는 만큼만 축소합니다.
3. **포지션 확대** — 투입된 잔고가 목표보다 작으면 목표를 향해 포지션을 늘립니다. 단, 손절로 인해 같은 방향의 재진입이 막혀 있으면(`pslReentry`) 늘리지 않습니다.
4. **정산** — 수수료, 실현 손익, 마진 이동은 심볼의 가격, 수량, quote 정밀도에 맞춰 반영됩니다. isolated 모드에서는 포지션을 열고 닫을 때 cross 잔고와 isolated 잔고 사이에서 마진이 이동하며, 시장가 진입 손실에 대비한 작은 여유분도 포함됩니다.

ATM-Eta와 동일한 할당 비율(지갑 잔고의 95%)을 적용하므로, 여기서 찾은 파라미터는 ATM-Eta에 배포했을 때도 일관되게 동작합니다.

<br>

#### 🔹 **단일 패스 잔고 추세 평가**

평가를 위해서는 모든 파라미터 세트의 성장률과 변동성이 필요하지만, 수만 개 레인의 전체 잔고 이력을 저장하는 것은 비용이 너무 큽니다. 대신 각 레인은 시뮬레이션 중에 로그 잔고에 대한 세 개의 누적 합을 `float64` 로 계산하고, 마지막에 닫힌 형태로 추세를 구합니다.

* **성장률** — 첫 거래 시점부터 시간에 대해 `ln(balance / initial balance)` 를 최소제곱법으로 적합한 직선의 기울기입니다. 인터벌당 평균 로그 성장률을 의미합니다.
* **변동성** — 해당 직선 대비 잔차의 표준편차입니다.

이 방식은 데이터 길이와 무관하게 레인당 메모리 비용을 일정하게 유지합니다. 전체 잔고 이력은 그래프가 필요한 `TEST` 와 `READ` 모드에서만 기록됩니다.

<br>

#### 🔹 **수치 정밀도**

`config.py` 의 `DATATYPE_PRECISION` 으로 시뮬레이션에 사용할 `float32` 와 `float64` 를 선택합니다. 탐색에는 `float32` 를 사용하며, `float64` 는 ATM-Eta의 CPU 기반 시뮬레이션 결과와 비교 검증하기 위한 용도입니다. 잔고 추세 누적값은 항상 `float64` 로 계산됩니다.

---



<a name="parameter-search"></a>
### 🔍 파라미터 탐색 ###

#### **파라미터 모델**

모든 후보는 모든 TEF 함수가 공유하는 **거래 파라미터** 와, 선택한 TEF 함수의 **모델 파라미터** 로 이루어진 벡터입니다.

| 그룹 | 파라미터 |
| :--- | :--- |
| 거래 | Full Stop Loss (Immediate), Full Stop Loss (Close) |
| 모델 | 각 TEF 함수의 `MODEL` 목록에서 정의 |

각 파라미터는 자신의 탐색 범위(`LIMIT`)와 소수점 정밀도(`PRECISION`)를 가집니다. 모든 후보는 이 정밀도로 양자화되므로, 탐색은 최종 Trade Configuration에서 사용할 것과 같은 이산 격자 위에서 이루어집니다. `tradeParamConfig` 와 `modelParamConfig` 로 원하는 파라미터를 상수로 고정하고, 나머지만 탐색하도록 할 수 있습니다.

<br>

#### **Seeker 알고리즘**

탐색은 각각 파라미터 공간의 한 점을 나타내는 **seeker** 들의 개체군으로 수행됩니다.

1. **수치 gradient** — 각 seeker의 각 파라미터마다, 해당 파라미터를 현재 값의 `deltaRatio` 만큼(최소 정밀도 한 단계) 위아래로 이동시킨 두 개의 테스트 지점을 만듭니다. `2 × nSeekers × nParameters` 개의 테스트 지점을 GPU 배치로 시뮬레이션하고, 점수의 중앙 차분으로 각 seeker의 gradient를 구합니다.
2. **Adam 방식 업데이트** — 각 seeker는 gradient의 지수이동평균(`beta_momentum`)과 gradient 제곱의 지수이동평균(`beta_velocity`)을 편향 보정하여 gradient 방향으로 이동합니다. 파라미터가 양자화되어 있으므로, 업데이트 후 반올림하면 같은 값으로 돌아가는 경우에는 정밀도 한 단계만큼 밀어내어 seeker가 멈추지 않도록 합니다.
3. **재배치 (자체 개발)** — `repopulationInterval` 단계마다, 점수가 가장 낮은 `repopulationRatio` 비율의 seeker들을 교체합니다. 교체되는 seeker 중 `repopulationGuideRatio` 비율은 살아남은 seeker들을 중심으로 한 정규분포에서 샘플링되며, 이 분포의 폭은 시간에 따라 좁아집니다(`repopulationDecayRate`). 나머지는 계속 탐색할 수 있도록 균등 분포로 무작위 배치됩니다.
4. **종료** — 최고 점수의 상대적 개선량을 `scoringSamples` 단계에 걸친 지수이동평균으로 추적합니다. 이 값이 `terminationThreshold` 아래로 떨어지면 현재 반복을 종료합니다.
5. **반복** — 전체 과정은 새로운 무작위 개체군으로 `nRepetition` 번 다시 시작되며, 모든 반복을 통틀어 가장 좋은 결과를 보관합니다.

<br>

#### **평가 기준**

| 평가 기준 | 공식 |
| :--- | :--- |
| `FINALBALANCE` | $1 - e^{-B_{final}/B_{initial}}$ |
| `GROWTHRATE` | $g \ge 0$ 이면 $S_{gr} = 1 + k_{gr}\,g$, 그 외에는 $1 / (1 - k_{gr}\,g)$ |
| `VOLATILITY` | $e^{-w_{vol}\,k_{vol}\,\sigma} \times (1 - e^{-k_{tv}\,V})^{w_{tv}}$ |
| `SHARPERATIO` | $S_{gr}^{\,w_{gr}} \times e^{-w_{vol}\,k_{vol}\,\sigma} \times (1 - e^{-k_{tv}\,V})^{w_{tv}}$ |

여기서 $g$ 는 성장률, $\sigma$ 는 변동성, $V$ 는 총 거래량입니다. $k$ 값들은 각 지표를 비교 가능한 범위로 맞추는 `scoring_*Scaler` 설정이고, $w$ 값들은 상대적 중요도를 정하는 `scoring_*Weight` 설정입니다. 거래량을 포함하는 이유는, 거래를 거의 하지 않는 파라미터 세트가 단순히 위험을 피하는 것만으로 높은 점수를 받지 못하게 하기 위해서입니다.

**최대 낙폭 필터** — 각 후보의 이론적인 99.7% 최악 낙폭을 $1 - e^{-3\sigma}$ 로 추정합니다. 이 값이 `scoring_maxMDD` 를 넘는 후보는 평가 기준과 관계없이 점수가 0이 됩니다.

---



### 🧩 TEF 함수 작성 ###

TEF 함수는 `teffunctions/` 폴더에 위치한 하나의 `teff_{NAME}.py` 파일입니다. 실행 시 `{NAME}` 이라는 이름으로 자동 등록되며, 이 이름을 `config.py` 의 `exitFunctionType` 에 사용합니다. 각 모듈은 다음을 정의해야 합니다.

| 정의 | 설명 |
| :--- | :--- |
| `MODEL` | 함수의 모델 파라미터 목록. 각 파라미터는 `PRECISION` 과 `LIMIT` 을 가짐 |
| `INPUTDATAKEYS` | 함수가 읽는 평탄화된 분석 컬럼의 키 목록 (접근 순서대로) |
| `PROCESSBATCH` | 공용 디스패처로 연결되는 배치 진입점 |
| `processBatch` | Triton 커널. 대부분은 공용 코드이며, 모델 파라미터, 상태 추적 변수, TEF 값 호출을 위한 표시된 섹션만 수정 |

TEF 값 계산 자체는 보통 별도의 `@triton.jit` 함수로 작성하며, 이 함수는 현재 행의 분석 데이터를 읽고, 모델의 상태 추적 변수를 갱신한 뒤, 방향과 TEF 값을 반환합니다. 새 TEF 함수를 작성할 때는 `teffunctions/` 폴더에 이미 포함된 모듈들(예: `teff_MMACDDEFAULT.py`)을 참고하시기 바랍니다. 기존 모듈을 복사한 뒤 표시된 섹션만 수정하는 것이 공용 시뮬레이션 엔진과의 호환성을 유지하는 가장 간단한 방법입니다.

TEFFP Seeker의 TEF 함수는 Triton 커널이고 ATM-Eta의 TEF 함수는 Python 함수이므로, 하나의 전략을 양쪽에 모두 구현해야 합니다. 인터페이스 계약은 동일하므로, 여기서 찾은 파라미터를 ATM-Eta 버전에 그대로 적용할 수 있습니다.

---



### 📤 출력 결과 ###

각 `SEEK` 실행은 `results/` 아래에 `teffps_result_{timestamp}` 폴더를 생성하며, 다음 파일을 포함합니다.

* **`{rCode}_result.json`** — 각 탐색 대상의 설정, 사용한 분석 데이터의 식별 정보(생성 시각, 시뮬레이션 코드, 심볼), 최적 결과, 그리고 탐색 중 최고 점수가 개선된 모든 기록.
* **`{rCode}_{index}_tc.tc`** — 각 탐색 대상의 최적 결과로 만든 ATM-Eta Trade Configuration:

```json
{
    "leverage":              5,
    "isolated":              true,
    "direction":             "BOTH",
    "orderType":             "MARKET",
    "orderOffset":           0.0,
    "fullStopLossImmediate": 0.0,
    "fullStopLossClose":     0.0075,
    "postStopLossReentry":   true,
    "teff_functionType":     "<TEF function name>",
    "teff_functionParams":   [ ... ]
}
```

`READ` 모드로 결과를 다시 읽을 때는 저장된 식별 정보를 현재 분석 데이터와 비교하므로, 결과가 다른 데이터셋으로 모르는 사이에 재평가되는 일은 없습니다.

---



### ⚠️ 시뮬레이션 범위 ###

이 시뮬레이터는 거래소를 완벽하게 재현하는 것이 아니라, 파라미터 탐색에 충분할 만큼 ATM-Eta의 실거래 동작에 가깝게 설계되었습니다. 구체적으로는 다음과 같습니다.

* 단순화를 위해 모든 주문은 인터벌 종가 기준 시장가 주문으로 시뮬레이션되며, 거래 수수료율(`tradingFee`)은 사용자가 조절할 수 있습니다. `LIMIT` 및 `ADAPTIVE` 주문 타입은 시뮬레이션하지 않으며, 내보내는 Trade Configuration은 `MARKET` 을 사용합니다.
* 슬리피지는 반영하지 않습니다.
* 인터벌 내부의 가격 경로는 알 수 없으므로, 같은 인터벌 안에서 발생한 청산들의 체결 순서는 근사적으로 처리됩니다.

모든 최적화 도구와 마찬가지로, 과거 데이터에서 찾은 최적 파라미터는 과적합되기 쉽습니다. 배포 전에 탐색 구간 밖의 데이터로 결과를 검증하는 것을 강력히 권장합니다.

---



### 🗓️ 프로젝트 기간
* 2024년 9월 – 2026년 3월 (이후 업데이트 및 유지보수 지속)

---

### 📄 문서 정보
**마지막 업데이트:** 2026년 9월 26일  
**작성자:** 김범수  
**이메일:**  kimlvis31@gmail.com 
