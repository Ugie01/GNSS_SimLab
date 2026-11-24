# GNSS SimLab — 위성항법(GNSS) · 항재밍(CRPA) · 통합항법(GNSS/INS EKF) 데스크톱 시뮬레이션 & 학습 플랫폼

임베디드 SW 엔지니어가 입사 전 위성항법(GNSS), 배열 신호처리 기반 항재밍(CRPA), GNSS-INS 통합항법(Error-State EKF)의 핵심 이론·수식·전문 용어·알고리즘을 **100% 로컬 오프라인 환경**에서 직관적으로 실습하고, 실전 C99/STM32/DSP/FPGA 포팅 노하우를 학습할 수 있는 Windows 설치형 데스크톱 애플리케이션입니다.

---

## 1. 핵심 설계 원칙 및 기술 스택

- **기술 스택**: Python 3.10+ / PySide6 / NumPy / SciPy / Matplotlib / Pillow / PyInstaller / Inno Setup 6
- **2D + 3D 하이브리드 대화형 시각화 스튜디오**:
  - 모든 모듈(`M01`~`M05`) 상단 헤더에 **`[📊 2D+3D 통합 뷰]`**, **`[🌐 3D 입체 회전 뷰]`**, **`[📈 2D 정밀 분석 뷰]`** 3단계 실시간 전환 스위치와 Matplotlib 인터랙티브 툴바(마우스 360° 입체 회전, 줌/팬, 고해상도 PNG/PDF 저장)를 기본 탑재했습니다.
  - **M01 3D**: WGS-84 지구 타원체 3D 메쉬 + 기준점 ECEF 위치 벡터 + 국소 ENU 접평면 및 직교 기저 벡터($\mathbf{e}_E, \mathbf{e}_N, \mathbf{e}_U$)
  - **M02 3D**: 2D FFT PCPS 신호 획득의 코드 위상($\tau$) × 도플러 주파수($f_d$) 3차원 상관 전력 곡면 $S(f_d, \tau)$
  - **M03 3D**: 상반구 위성 시선 벡터(LOS) 3D 돔 + 가시/차폐 위성 광선 + 3차원 DOP 오차 타원체($\mathrm{EDOP}, \mathrm{NDOP}, \mathrm{VDOP}$)
  - **M04 3D**: CRPA MVDR 빔포머의 상반구 3차원 방향성 방사패턴 $G(\theta, \phi)$ 및 재머 방향 3D 공간 널(Null) 협곡
  - **M05 3D**: UAV/드론의 3차원 입체 기동 궤적($[\text{East}, \text{North}, \text{Up}]$) + 순수 INS 3D 표류 vs Error-State EKF 재밍 Outage 관성 주행
- **100% 로컬 오프라인 동작**: 외부 CDN이나 인터넷 연결, 시스템 TeX 설치 없이 Matplotlib 내장 `mathtext` 엔진을 이용해 모든 수식을 고해상도 투명 `QPixmap`으로 선명하게 렌더링합니다.
- **엄격한 계층 분리 (`core/` vs `ui/` vs `modules/`)**:
  - `core/`: UI 의존성이 전혀 없는 순수 NumPy/SciPy 수학·물리 엔진 (`pytest` 독립 검증)
  - `ui/`: 추상 기본 클래스 `BaseLabModule`, 모듈 레지스트리 `ModuleRegistry`, 수식 카드 `FormulaCardWidget`, 인터랙티브 용어 사전(`glossary://<term_id>`)
  - `modules/`: 플러그인 형태의 학습 모듈 5종 (`M01`~`M05`) + 신규 모듈 확장 템플릿 (`_template_new_module.py`)

---

## 2. 전체 학습 커리큘럼 (5대 핵심 모듈 + 40개+ 용어 사전)

| 모듈 ID | 모듈명 | 핵심 물리/수학 모델 (`core/`) | 임베디드 C 포팅 핵심 팁 |
| :--- | :--- | :--- | :--- |
| **MOD 01** | **좌표계 변환 · float32/64 정밀도 · GPST/1PPS** | WGS-84 LLH ↔ ECEF ↔ ENU 변환, 묘유선 곡률반경 $N(\phi)$, IEEE 754 `float32` 50cm 양자화 계단 오차, GPST Week/TOW 및 1PPS vs UART 지연/클럭 드리프트 | 절대 좌표(`ECEF`/`LLH`) `double` 필수 및 상대 변위(`ΔENU`) `float` 변환, STM32 `TIM2 Input Capture` 1PPS 하드웨어 시각 동기 |
| **MOD 02** | **PRN 골드코드 상관 · 2D FFT 획득 · DLL/PLL 추적** | IS-GPS-200 G1/G2 LFSR 기반 PRN 1~32 Gold Code 생성, 원형 자기/상호상관, 2D FFT 병렬 코드위상 탐색(PCPS), Early-Minus-Late DLL S-Curve, 2차 FLL-assisted Costas PLL(50bps 항법비트 복조) | 32비트 부호 없는 정수 위상 누산기(`uint32_t` Modulo $2^{32}$) 기반 NCO, 1ms 상관 덤프 ISR 루프 필터 및 Carrier-Aiding |
| **MOD 03** | **Skyplot · DOP · 가우스-뉴턴 WLS 측위 · SBAS** | 11기 위성 궤도 및 도심 빌딩 협곡 차폐, 기하 행렬 $H$ 및 7종 DOP(GDOP/PDOP/HDOP/VDOP/TDOP/EDOP/NDOP), 전리층 경사계수·대류권·다중경로 오차, 가우스-뉴턴 WLS 반복 수렴 및 KASS/SBAS 보정 | 동적 할당(`malloc`) 없는 4×4 정규방정식($H^T W H$) 직접 누적 패턴, 전파 비행시간 동안의 지구 자전(Sagnac 효과) 보정 |
| **MOD 04** | **항재밍 CRPA 배열 안테나 · Power-Min / MVDR 널링** | $N$소자 균일 선형 배열(ULA) 조향 벡터 $\mathbf{a}(\theta)$, 다중 재머 공간 공분산 행렬 $\hat{\mathbf{R}}_{xx}$ 및 대각 로딩, 전력 최소화(Power-Min/PI) 및 MVDR(Capon) 적응형 빔포밍, 고유값 부공간 및 유효 $C/N_0$ 복원 | 에르미트 양정치 공분산 행렬의 4×4 복소 Cholesky 분해($\mathbf{R} = \mathbf{L}\mathbf{L}^H$) 및 전방/후방 대입 기반 역행렬 없는 가중치 산출 |
| **MOD 05** | **GNSS-INS 통합항법 · Error-State EKF · 재밍 Outage** | 50Hz IMU / 5Hz GNSS Figure-8 궤적 및 동체 비력 생성, 순수 INS 추측항법 표류, 8상태 오차 칼만필터(ESKF) 온라인 가속도/자이로 바이어스 추정, GNSS Outage(재밍) 관성 주행 및 복구 | 조셉 형태(Joseph Form) 수치 안정화 공분산 갱신 및 대칭화 강제, 방위각 래핑(`WrapPi`), 오차 상태 주입(Injection) 및 리셋 |
| **MOD 06** | **[항법연구소 특화] EGI 실무 · RAIM/항기만 · STAP/레버암** | RAIM 카이제곱($\chi^2$) 무결성 검정 및 FDE 고장 위성 자동 배제, Pull-off 스푸핑 상관 피크 납치 및 SQM 비대칭 탐지, 고기동 레버암($\mathbf{C}_b^n\mathbf{l}^b$)·타임태그 지연 보정, 다중 탭 STAP 광대역 널링 | 유도무기/무인기 EGI RTOS 5-태스크 아키텍처, 스트랩다운 INS 2-샘플 코닝(Coning) & 스컬링(Sculling) 보정, 1PPS 링버퍼 매칭 및 카이제곱 게이팅 |

---

## 3. 프로젝트 디렉토리 구조

```text
GNSS_SimLab/
├── main.py                         # 데스크톱 앱 엔트리포인트
├── requirements.txt                # 버전 호환성이 검증된 패키지 의존성 목록
├── pytest.ini                      # pytest 경로 및 테스트 설정
├── README.md                       # 아키텍처, 빌드 및 모듈 확장 가이드
├── assets/
│   └── app.ico                     # 다중 해상도(16~256px) 위성/궤도/신호 테마 아이콘
├── core/                           # UI 의존성 없는 순수 NumPy/SciPy 수학·물리 엔진
│   ├── __init__.py
│   ├── resource_helper.py          # 개발 환경 및 PyInstaller(_MEIPASS) 겸용 경로 헬퍼
│   ├── coords.py                   # Module 1 코어 엔진
│   ├── signal_proc.py              # Module 2 코어 엔진
│   ├── pvt_dop.py                  # Module 3 코어 엔진
│   ├── antijam_crpa.py             # Module 4 코어 엔진
│   └── gnss_ins_ekf.py             # Module 5 코어 엔진
├── data/
│   ├── glossary.json               # 40개 전문 용어 사전 데이터 (한/영/요약/해설/수식/연관모듈)
│   └── presets.json                # 5개 모듈 × 4개 시나리오 = 총 20개 실전 시나리오 프리셋
├── ui/                             # 공통 UI 프레임워크
│   ├── __init__.py
│   ├── base_module.py              # 추상 기본 클래스 BaseLabModule
│   ├── registry.py                 # 중앙 모듈 등록 레지스트리 (ModuleRegistry)
│   ├── formula_widget.py           # Matplotlib mathtext 고해상도 수식 + 기호 해설 표 위젯
│   ├── glossary_dialog.py          # glossary://<id> 클릭 시 즉시 뜨는 용어 해설 팝업
│   ├── glossary_tab.py             # 실시간 키워드 검색 & 분야 필터링 용어 사전 탭
│   └── main_window.py              # 메인 윈도우 (다크 사이드바 + 워크스페이스 스택)
├── modules/                        # 플러그인 학습 모듈
│   ├── __init__.py
│   ├── m01_coords_time.py
│   ├── m02_signal_acq_track.py
│   ├── m03_pvt_dop.py
│   ├── m04_antijam_crpa.py
│   ├── m05_gnss_ins_ekf.py
│   └── _template_new_module.py     # 신규 모듈(RTK, 스푸핑, STM32 시리얼 등) 확장용 표준 템플릿
├── tests/                          # 수학적 정확성 및 UI 통합 검증용 pytest 스위트 (33개 테스트)
│   ├── test_coords.py
│   ├── test_signal_proc.py
│   ├── test_pvt_dop.py
│   ├── test_antijam_crpa.py
│   ├── test_gnss_ins_ekf.py
│   └── test_ui_integration.py
├── scripts/
│   ├── generate_icon.py            # Pillow 기반 멀티 해상도 아이콘 자동 생성기
│   └── build_windows.bat           # 테스트 -> 아이콘 -> PyInstaller -> Inno Setup 원클릭 빌드
└── packaging/
    ├── gnss_simlab.spec            # PyInstaller 번들링 스펙 (--noconsole)
    └── installer.iss               # Inno Setup 6 설치/언인스톨러(unins000.exe) 생성 스크립트
```

---

## 4. 개발 환경 실행 및 원클릭 Windows 설치 파일 빌드

### 4.1 가상환경 활성화 및 앱 실행
```powershell
cd "c:\Users\dbdna\OneDrive\바탕 화면\취미\GNSS_SimLab"
.\.venv\Scripts\python.exe main.py
```

### 4.2 전체 단위/통합 테스트 실행 (33개 항목)
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

### 4.3 원클릭 EXE 및 Windows 설치 프로그램(`Setup.exe`) 빌드
```powershell
.\scripts\build_windows.bat
```
- **독립 실행파일 출력**: `dist\GNSS_SimLab\GNSS_SimLab.exe`
- **Windows 설치 파일 출력**: `dist_installer\GNSS_SimLab_Setup.exe` (바탕화면/시작메뉴 바로가기, 커스텀 아이콘, 표준 제어판 제거 프로그램 `unins000.exe` 자동 등록)

---

## 5. 새 모듈(예: RTK, 스푸핑, STM32 시리얼 연동) 1줄 확장 가이드

기존 UI 프레임워크 코드를 전혀 수정하지 않고 새 모듈을 추가할 수 있습니다:

1. **템플릿 복사**: `modules/_template_new_module.py`를 `modules/m06_rtk_lambda.py`로 복사합니다.
2. **클래스 구현**: `BaseLabModule`을 상속받아 파라미터 위젯(`build_parameter_widget`), 그래프(`run_simulation`), 이론 HTML(`get_theory_html`), 수식 리스트(`get_formula_specs`), 관찰 가이드(`get_observation_guide_html`), 임베디드 C 포팅 팁(`get_embedded_c_porting_html`)을 작성합니다.
3. **레지스트리 1줄 등록**: `ui/registry.py`의 `register_all_modules()`에 아래 한 줄만 추가하면 좌측 사이드바 메뉴와 프리셋 시스템에 즉시 연동됩니다.
   ```python
   from modules.m06_rtk_lambda import Module06RtkLambda
   ModuleRegistry.register(Module06RtkLambda)
   ```
