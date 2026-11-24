"""
Pure Python / NumPy curriculum scheduling and persistence engine for the GNSS SimLab Study Planner.
Generates an adaptive day-by-day evening study schedule (default 2.0 hours/day) for any user-specified
number of study days (e.g. 3 to 60 days), supports full user editing, progress tracking, and JSON persistence.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.resource_helper import resource_path


@dataclass
class StudyDayItem:
    """Editable data model for a single day's study session."""

    day_number: int
    date_str: str
    phase_badge: str
    title: str
    target_module_id: str
    target_level: str  # "lv1", "both", "lv2"
    recommended_preset_idx: int
    hours: float
    step1_concept: str
    step2_practice: str
    step3_review_c: str
    self_check_question: str
    user_notes: str = ""
    completed: bool = False


@dataclass
class StudyPlanDocument:
    """Container for the entire user study plan, metadata, and daily items."""

    start_date_str: str
    total_days: int
    daily_hours: float
    focus_track: str  # "balanced", "basic_focus", "rnd_deep"
    days: List[StudyDayItem] = field(default_factory=list)


# ==============================================================================
# Master 16-Unit Step-by-Step Pedagogical Curriculum Pool
# (Ordered from Basic GNSS Lv.1 Intuition -> Lv.2 Math -> Danam R&D CRPA/EGI/C99)
# ==============================================================================
MASTER_CURRICULUM_UNITS: List[Dict[str, Any]] = [
    {
        "phase": "1단계: 위성항법 기초 (Lv.1 입문)",
        "title": "좌표계 3총사(LLH·ECEF·ENU)와 3D 지구 타원체 직관 잡기",
        "module_id": "m01_coords_time",
        "level": "lv1",
        "preset_idx": 0,
        "step1": "M01 [Lv.1 쉬운 입문] 탭을 켜고 도로명 주소(LLH), 우주 본부 좌표(ECEF), 내 발밑 돗자리(ENU)의 차이점과 WGS-84 타원체 개념을 읽습니다.",
        "step2": "[2D 핵심 그래프] 및 [3D 입체 시각화] 탭에서 서울 시청과 적도(Quito) 프리셋을 번갈아 선택하며 3D 지구본 위의 ENU 빨강·초록·파랑 화살표 방향을 관찰합니다.",
        "step3": "[전문 용어 사전]에서 WGS-84, LLH, ECEF, ENU, NED, 묘유선 곡률반경 6개 용어의 Lv.1 요약을 정독합니다.",
        "check_q": "왜 위성 궤도 계산은 ECEF에서 하고, 드론/미사일 제어는 ENU(또는 NED)에서 하는지 1문장으로 설명할 수 있는가?",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.1→2)",
        "title": "임베디드 float32 50cm 계단 오차와 GPS 시간(GPST) · 1PPS 동기",
        "module_id": "m01_coords_time",
        "level": "both",
        "preset_idx": 1,
        "step1": "M01 [Lv.2 심화 · 수식] 탭에서 왜 32비트 float가 637만m 크기의 ECEF 좌표를 담을 때 50cm 단위로 뚝뚝 끊기는지(ULP 수식)와 윤초 없는 GPST를 학습합니다.",
        "step2": "원형 궤적 반경 슬라이더를 0.5m~2.0m로 줄여 파란 원(double) vs 빨간 계단(float)을 비교하고, [그래프 3 · 4 크게 보기]에서 UART 지연(450ms) vs 1PPS 하드웨어 핀 타이밍도를 실험합니다.",
        "step3": "[실전 C99 코드] 탭에서 double ECEF 뺄셈 후 float ENU로 내리는 C99 코드와 STM32 TIM2 Input Capture 1PPS 인터럽트 코드를 분석합니다.",
        "check_q": "UART로 들어오는 NMEA 시간만 믿으면 왜 수백 ms 오차가 생기며, 1PPS 핀과 어떻게 조합해야 정확한 시각을 얻는가?",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.1 입문)",
        "title": "열잡음 아래 숨은 위성 찾기: CDMA 대역확산과 PRN 골드코드",
        "module_id": "m02_signal_acq_track",
        "level": "lv1",
        "preset_idx": 0,
        "step1": "M02 [Lv.1 쉬운 입문] 탭에서 열잡음(-110dBm)보다 100배 약한 GPS 신호(-130dBm)를 1023칩 비밀 톱니바퀴(PRN 코드)로 건져 올리는 원리를 이해합니다.",
        "step2": "2D 그래프 1에서 목표 위성(PRN #1)과 간섭 위성(PRN #7)을 비교하고, 두 번호를 똑같이(#7 vs #7) 맞춰보며 자기상관 삼각 피크와 상호상관 억제를 확인합니다.",
        "step3": "용어 사전에서 PRN Code, CDMA, C/N0(반송파 대 잡음밀도 비), 도플러 편이 4개 용어를 복습합니다.",
        "check_q": "모든 GPS 위성이 똑같은 주파수(1575.42 MHz)를 동시에 쏘는데도 내 수신기가 1번 위성과 7번 위성을 섞이지 않게 구분하는 원리는?",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.1→2)",
        "title": "2D FFT 병렬 코드위상 탐색(PCPS) 신호 획득과 동기 적분 시간",
        "module_id": "m02_signal_acq_track",
        "level": "both",
        "preset_idx": 3,
        "step1": "M02 [Lv.2 심화 · 수식] 수식 1을 통해 시간 영역에서 1023번 슬라이딩하는 대신 FFT 주파수 영역 곱셈 한 번으로 모든 코드 지연을 찾는 PCPS 수식을 익힙니다.",
        "step2": "신호 품질(C/N0)을 28 dB-Hz로 낮춰 획득 실패를 확인한 뒤, 동기 적분 시간(T_coh)을 1ms → 10ms로 늘려 [3D 입체 시각화] 탭 상관 평면에서 피크가 솟아오르는 것을 관찰합니다.",
        "step3": "도플러 단면(Inset 그래프)에서 위성 속도에 의해 반송파 주파수가 수 kHz 밀리는 현상과 탐색 빈(Bin) 간격의 의미를 노트에 정리합니다.",
        "check_q": "약신호 환경에서 동기 적분 시간(T_coh)을 1ms에서 10ms로 늘리면 상관 피크 SNR은 몇 dB 좋아지며, 왜 무한정(예: 100ms) 늘릴 수는 없는가?(힌트: 20ms 항법비트 반전)",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.1→2)",
        "title": "정밀 신호 추적: Early-Late DLL(S-Curve)과 Costas PLL 항법비트 복조",
        "module_id": "m02_signal_acq_track",
        "level": "both",
        "preset_idx": 1,
        "step1": "M02 수식 2·3을 통해 양쪽 어깨 높이를 맞추는 DLL(Early-Prompt-Late)과 180° 위상 반전에 둔감한 코스타스 루프(atan(Q_P/I_P))의 역할을 학습합니다.",
        "step2": "E-L 상관기 간격(d)을 1.0 → 0.2 chip(Narrow Correlator)으로 줄여 거리 지터(σ_DLL) 감소를 확인하고, [그래프 3 · 4 크게 보기]에서 Q_P가 0으로 수렴하며 I_P에서 50bps 비트가 나오는 것을 관찰합니다.",
        "step3": "[실전 C99 코드] 탭에서 32비트 정수 오버플로우(uint32_t)를 이용한 NCO 위상 누산기와 1ms 상관 덤프 ISR 루프 필터 C 코드를 학습합니다.",
        "check_q": "일반 PLL(atan2(Q, I)) 대신 코스타스 PLL(atan(Q/I))을 써야만 하는 이유가 50 bps 항법 데이터 비트와 어떤 관계가 있는가?",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.1 입문)",
        "title": "의사거리(Pseudorange)와 최소 위성 4기 조건 · Skyplot · DOP 기하학",
        "module_id": "m03_pvt_dop",
        "level": "lv1",
        "preset_idx": 0,
        "step1": "M03 [Lv.1 쉬운 입문] 탭에서 왜 미지수가 (X, Y, Z, 수신기 시계오차 b) 4개인지, 그리고 왜 땅 밑에는 위성이 없어 항상 수직 오차(VDOP)가 수평(HDOP)보다 나쁜지 이해합니다.",
        "step2": "도심 빌딩 차폐 슬라이더를 0° → 52°로 올려 동·서쪽 위성이 가려질 때 EDOP(동서 오차배율)가 치솟고 [3D 입체 시각화] 탭의 빨간 오차 타원체가 옆으로 늘어나는 것을 확인합니다.",
        "step3": "앙각 마스크를 48° 이상으로 올려 가시 위성이 3기 이하로 떨어질 때 왜 측위 불능(Rank < 4) 경고가 뜨는지 직접 확인합니다.",
        "check_q": "빛은 1 마이크로초(1 μs)에 약 300m를 진행한다. 수신기 시계가 불과 1 밀리초(1 ms)만 틀어져도 의사거리는 몇 km나 틀어지는가?",
    },
    {
        "phase": "1단계: 위성항법 기초 (Lv.2 실무)",
        "title": "가우스-뉴턴 WLS 반복 수렴 · 전리층/대류권 오차 · KASS(SBAS) 보정",
        "module_id": "m03_pvt_dop",
        "level": "both",
        "preset_idx": 2,
        "step1": "M03 [Lv.2 심화 · 수식] 수식 1·2·3을 통해 비선형 거리 방정식을 시선 벡터 행렬 H로 선형화하여 지구 중심(0,0,0)에서 5~6회 만에 지표면으로 수렴시키는 WLS를 학습합니다.",
        "step2": "앙각 가중치(W_ii = sin²El) 체크박스를 껐다 켜며 저앙각 위성 오차 억제 효과를 비교하고, 2D 그래프 4에서 단독 측위(빨간 점) vs SBAS 보정(초록 점) 오차를 비교합니다.",
        "step3": "[실전 C99 코드] 탭에서 동적 할당(malloc) 없이 4x4 정규방정식에 직접 누적하는 C99 코드와 지구 자전(Sagnac 효과, ~25m) 보정 공식을 숙지합니다.",
        "check_q": "고도가 낮은(저앙각) 위성일수록 전리층·대류권 지연과 다중경로 오차가 급증하는 이유와, 이를 WLS 가중치 행렬 W에 어떻게 반영하는가?",
    },
    {
        "phase": "2단계: 항재밍 · 복합항법 (Lv.1 입문)",
        "title": "CRPA 배열 안테나 공간 필터링(Nulling)과 공간 자유도(N-1)의 원리",
        "module_id": "m04_antijam_crpa",
        "level": "lv1",
        "preset_idx": 0,
        "step1": "M04 [Lv.1 쉬운 입문] 탭에서 반파장(9.5cm) 간격 배열 안테나가 위상차를 이용해 재머 방향의 귀만 -50dB 골짜기(Null)로 막아버리는 노이즈 캔슬링 원리를 이해합니다.",
        "step2": "프리셋 1(4소자)과 프리셋 4(2소자 자유도 초과)를 비교하며, 안테나 N개가 최대 N-1개의 재머만 막을 수 있다는 자유도 한계를 2D 그래프 1과 [3D 입체 시각화] 탭으로 확인합니다.",
        "step3": "소자 간격(d/λ)을 0.50λ → 0.95λ로 늘려 엉뚱한 각도에 가짜 널(Grating Lobe / Spatial Aliasing)이 생기는 이유를 관찰합니다.",
        "check_q": "4소자 CRPA 배열 안테나가 독립적인 방향에서 들어오는 강한 재머를 동시에 최대 몇 개까지 널링(Nulling)할 수 있는가?",
    },
    {
        "phase": "2단계: 항재밍 · 복합항법 (Lv.2 실무)",
        "title": "공간 공분산 행렬(R_xx)과 Power-Min(PI) vs MVDR(Capon) 빔포머 · C 포팅",
        "module_id": "m04_antijam_crpa",
        "level": "both",
        "preset_idx": 1,
        "step1": "M04 [Lv.2 심화 · 수식] 수식 1·2·3을 통해 위성 방향을 몰라도 재머를 깎는 Power-Min과 위성 방향 이득을 0dB로 보존하는 MVDR 수식을 비교합니다.",
        "step2": "7소자 프리셋과 위성 인접 재밍(+15° 위성 옆 +22° 재머) 프리셋을 실험하고, 2D 그래프 3의 공분산 고유값(Eigenvalue) 막대 개수가 재머 개수와 일치함을 확인합니다.",
        "step3": "[실전 C99 코드] 탭에서 대각 로딩(Diagonal Loading)과 4x4 복소 Cholesky 분해(R = L·L^H) 전방/후방 대입 C99 코드를 분석합니다.",
        "check_q": "Power-Min(전력 최소화) 기법은 위성이 어느 방향에 있는지 전혀 몰라도 왜 위성 신호는 안 깎고 재머 신호만 골라서 널링할 수 있는가?",
    },
    {
        "phase": "2단계: 항재밍 · 복합항법 (Lv.1 입문)",
        "title": "왜 GNSS와 IMU는 환상의 짝꿍인가? 순수 INS 표류와 바이어스 사전 학습",
        "module_id": "m05_gnss_ins_ekf",
        "level": "lv1",
        "preset_idx": 0,
        "step1": "M05 [Lv.1 쉬운 입문] 탭에서 '눈 감고 걸음 수 세기(IMU 추측항법)'와 '등대(GPS)' 비유를 읽고, 왜 GPS가 끊기기 전에 센서 영점(바이어스)을 미리 고쳐두어야 하는지 이해합니다.",
        "step2": "프리셋 1(22초에 재밍 진입: 바이어스 학습 완료)과 프리셋 3(3초 만에 조기 재밍 진입: 바이어스 미학습)을 번갈아 눌러 재밍 구간(빨간 굵은 선)의 표류 차이를 비교합니다.",
        "step3": "[3D 입체 시각화] 탭에서 드론의 3D 8자 비행 궤적과 순수 INS의 3D 발산 궤적, 그리고 2D 그래프 2의 초록 점선(3σ 불확실성 풍선)이 재밍 중 부풀었다가 복구 시 수축하는 것을 관찰합니다.",
        "check_q": "출발 직후 GPS가 잘 터지는 구간에서 칼만 필터가 미리 추정해 두는 가장 중요한 IMU 센서 오차 2가지는 무엇인가?",
    },
    {
        "phase": "2단계: 항재밍 · 복합항법 (Lv.2 실무)",
        "title": "8상태 Error-State EKF 동역학과 조셉 형태(Joseph Form) 공분산 안정화",
        "module_id": "m05_gnss_ins_ekf",
        "level": "both",
        "preset_idx": 1,
        "step1": "M05 [Lv.2 심화 · 수식] 수식 1·2·3을 통해 왜 전체 상태(Total State) 대신 오차 상태(Error State: δp, δv, δψ, δb_a, δb_g)를 추정하고 주입(Injection) 후 0으로 리셋하는지 학습합니다.",
        "step2": "시스템 잡음(Q)과 측정 잡음(R) 배율 슬라이더를 조절하며 [그래프 3 · 4 크게 보기]에서 자이로/가속도 바이어스 수렴 속도와 기수 방위각(Heading) 오차 변화를 실험합니다.",
        "step3": "[실전 C99 코드] 탭에서 부동소수점 비대칭 발산(NaN)을 막는 조셉 형태(Joseph Form) 공분산 갱신과 각도 래핑(WrapPi) C99 코드를 숙지합니다.",
        "check_q": "임베디드 C에서 단순 공분산 갱신식 P = (I - KH)P 대신 조셉 형태를 사용하고 대칭화 루프를 돌려야 하는 수치적 이유는?",
    },
    {
        "phase": "3단계: 단암 항법연구소 특화 (Lv.1→2)",
        "title": "RAIM 자율 무결성 감시 · FDE 고장 위성 배제 · Pull-Off 항기만(SQM)",
        "module_id": "m06_egi_raim_spoof",
        "level": "both",
        "preset_idx": 0,
        "step1": "M06 [Lv.2 심화 · 수식] 탭에서 5기 이상 위성 잔차로 고장을 감지하는 RAIM(카이제곱 WSSE)과 6기 이상에서 범인 위성을 빼버리는 FDE, 그리고 상관 피크 납치(Pull-off Spoofing)를 학습합니다.",
        "step2": "G12 오차 주입 슬라이더를 0m → 60m로 올려 2D 그래프 1에서 G12(빨간 막대)가 자동 배제되며 위치 오차가 20m → 1.4m로 복구되는 것과, 2D 그래프 2 · [3D 입체 시각화] 탭의 스푸핑 피크 분리를 관찰합니다.",
        "step3": "기만 코드 오프셋을 0.25~0.45 chip으로 두었을 때 상관 삼각형의 좌우 대칭이 깨지며 SQM 비대칭 경보(M_Δ)가 울리는 메커니즘을 정리합니다.",
        "check_q": "재밍(Jamming)과 스푸핑(Spoofing)의 차이점은 무엇이며, 스푸퍼가 추적 루프를 납치할 때 SQM과 RAIM은 각각 어떻게 이를 탐지하는가?",
    },
    {
        "phase": "3단계: 단암 항법연구소 특화 (Lv.2 실무)",
        "title": "유도무기/무인기 EGI 실무: 레버암 · 타임태그 링버퍼 · 코닝/스컬링 · STAP",
        "module_id": "m06_egi_raim_spoof",
        "level": "both",
        "preset_idx": 3,
        "step1": "M06 수식 3과 [실전 C99 코드] 탭의 RTOS 5-태스크 아키텍처 표(1PPS ISR, 1kHz 상관기 ISR, 400Hz INS 태스크, 100Hz CRPA/STAP, 10Hz EGI 칼만 태스크)를 정독합니다.",
        "step2": "[그래프 3 · 4 크게 보기]에서 선회 기동 시 레버암(1.2m)과 통신 지연(40ms) 때문에 생기는 가짜 오차가 링버퍼 매칭으로 사라지는 것과, 다중 탭 STAP 광대역 널 깊이를 실험합니다.",
        "step3": "고주파 진동 환경 필수 코드인 2-샘플 코닝(Coning) · 스컬링(Sculling) 외적 보정 C99 코드와 마할라노비스 거리(카이제곱 게이팅) 기각 코드를 완벽히 이해합니다.",
        "check_q": "고속 비행체에 장착된 IMU와 GPS 안테나 위치가 1.5m 떨어져 있고(레버암) GPS 데이터가 50ms 늦게 도착할 때, 칼만 필터에서 이를 어떻게 시간·공간적으로 일치시켜 비교하는가?",
    },
    {
        "phase": "4단계: 입사 전 최종 완성",
        "title": "방산 항법 핵심 용어 50선 스피드 마스터 · 전 모듈(M01~M06) 총정리 리허설",
        "module_id": "m06_egi_raim_spoof",
        "level": "lv2",
        "preset_idx": 0,
        "step1": "좌측 사이드바 하단 [전문 용어 사전 (50개)] 탭을 열고 분야별(좌표계·신호처리·측위·항재밍·통합항법·임베디드)로 용어를 클릭하며 Lv.1 직관과 Lv.2 수식을 복습합니다.",
        "step2": "M01부터 M06까지 [Lv.2 심화 · 수식] · [실전 C99 코드] 탭을 훑어보며 막히는 용어나 수식이 없는지 최종 점검합니다.",
        "step3": "내 학습 노트에 입사 후 사수 및 팀원들과 대화할 핵심 키워드(ECEF double, 1PPS 캡처, PCPS, Costas PLL, GDOP, Power-Min vs MVDR, STAP, ESKF, 레버암/코닝·스컬링, RAIM/FDE)를 정리합니다.",
        "check_q": "단암시스템즈 항법기술연구소 SW팀이 개발하는 항재밍(CRPA/STAP) 신호처리부터 EGI(GNSS/INS EKF + RAIM)까지의 전체 데이터 흐름을 머릿속으로 그릴 수 있는가?",
    },
]


def get_user_study_plan_path() -> Path:
    """
    Return a writable file path for storing the user's customized study plan JSON.
    In development mode, uses `data/user_study_plan.json`.
    If running inside a read-only Program Files installation, falls back to `%LOCALAPPDATA%/GNSS_SimLab/user_study_plan.json`.
    """
    local_data_path = Path(resource_path("data/user_study_plan.json"))
    try:
        local_data_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write permission on parent directory
        if os.access(str(local_data_path.parent), os.W_OK):
            return local_data_path
    except OSError:
        pass

    appdata = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    fallback_dir = Path(appdata) / "GNSS_SimLab"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir / "user_study_plan.json"


def generate_study_plan(
    total_days: int = 14,
    start_date: date | None = None,
    daily_hours: float = 2.0,
    focus_track: str = "balanced",
) -> StudyPlanDocument:
    """
    Dynamically generate a custom step-by-step evening study plan tailored to `total_days`
    (from 1 to 90 days) and `daily_hours` (default 2.0 hours/day).

    - If `focus_track == "basic_focus"`: Prioritizes M01~M03 basic GNSS foundation + Lv.1 intuition of M04~M06.
    - If `focus_track == "rnd_deep"`: Prioritizes Lv.2 equations, C99 porting, and Danam R&D M04~M06.
    - If `focus_track == "balanced"`: Full step-by-step progression from Lv.1 basic GNSS to Lv.2 Danam R&D.
    """
    total_days = int(np.clip(total_days, 1, 90))
    daily_hours = float(np.clip(daily_hours, 0.5, 12.0))
    if start_date is None:
        start_date = date.today()

    # Select and weight curriculum units according to focus_track
    units = list(MASTER_CURRICULUM_UNITS)
    if focus_track == "basic_focus":
        # Emphasize first 10 units (M01~M05 basics) + summary
        selected_pool = units[:11] + [units[11], units[13]]
    else:
        selected_pool = units

    n_units = len(selected_pool)
    day_items: List[StudyDayItem] = []

    min_total = int(round(daily_hours * 60))
    m1 = int(round(min_total * 0.33))  # ~40 min for 2h
    m2 = int(round(min_total * 0.42))  # ~50 min for 2h
    m3 = max(min_total - m1 - m2, 10)  # ~30 min for 2h

    if total_days <= n_units:
        # Partition `n_units` evenly across `total_days` buckets (merging adjacent units when compressed)
        splits = np.array_split(np.arange(n_units), total_days)
        for d_idx, idx_arr in enumerate(splits):
            cur_date = start_date + timedelta(days=d_idx)
            chunk = [selected_pool[int(i)] for i in idx_arr]
            primary = chunk[0]
            if len(chunk) == 1:
                title = primary["title"]
                s1 = f"[1교시 {m1}분 · 개념] {primary['step1']}"
                s2 = f"[2교시 {m2}분 · 실습] {primary['step2']}"
                s3 = f"[3교시 {m3}분 · 정리] {primary['step3']}"
                q_txt = primary["check_q"]
            else:
                titles = " + ".join(u["title"].split(":")[0] if ":" in u["title"] else u["title"][:18] for u in chunk)
                title = f"[압축 핵심] {titles}"
                s1 = f"[1교시 {m1}분 · 개념] " + " / ".join(u["step1"] for u in chunk)
                s2 = f"[2교시 {m2}분 · 실습] " + " / ".join(u["step2"] for u in chunk)
                s3 = f"[3교시 {m3}분 · 정리] " + " / ".join(u["step3"] for u in chunk)
                q_txt = " | ".join(u["check_q"] for u in chunk)

            lvl = "lv1" if focus_track == "basic_focus" and d_idx < total_days // 2 else (
                "lv2" if focus_track == "rnd_deep" and d_idx > 1 else primary["level"]
            )

            day_items.append(
                StudyDayItem(
                    day_number=d_idx + 1,
                    date_str=cur_date.strftime("%Y-%m-%d"),
                    phase_badge=primary["phase"],
                    title=title,
                    target_module_id=primary["module_id"],
                    target_level=lvl,
                    recommended_preset_idx=int(primary["preset_idx"]),
                    hours=daily_hours,
                    step1_concept=s1,
                    step2_practice=s2,
                    step3_review_c=s3,
                    self_check_question=q_txt,
                    user_notes="",
                    completed=False,
                )
            )
    else:
        # Expand `n_units` across `total_days` by splitting units into:
        # Day A: 🌱 Lv.1 입문 비유 & 시뮬레이션 그래프 직관 실습
        # Day B: 🚀 Lv.2 연구소 수식 심화 & 임베디드 C99 코드 분석 / 복습
        for d_idx in range(total_days):
            cur_date = start_date + timedelta(days=d_idx)
            # Map day index smoothly to unit index
            u_idx = int((d_idx * n_units) // total_days)
            prev_u_idx = int(((d_idx - 1) * n_units) // total_days) if d_idx > 0 else -1
            unit = selected_pool[min(u_idx, n_units - 1)]

            is_deep_followup = (u_idx == prev_u_idx)
            if not is_deep_followup:
                lvl = "lv1" if unit["level"] == "lv1" else "both"
                title = f"[개념 · 직관] {unit['title']}"
                s1 = f"[1교시 {m1}분 · Lv.1 비유] {unit['step1']}"
                s2 = f"[2교시 {m2}분 · 2D/3D 실습] {unit['step2']}"
                s3 = f"[3교시 {m3}분 · 용어 정리] 사수의 실시간 쉬운 해설을 읽고 핵심 용어를 내 말로 요약합니다."
            else:
                lvl = "lv2"
                title = f"[수식 · C포팅 심화] {unit['title']}"
                s1 = f"[1교시 {m1}분 · Lv.2 수식] 해당 모듈의 [Lv.2 심화 · 수식] 탭에서 수식 카드의 모든 기호와 단위를 분석합니다."
                s2 = f"[2교시 {m2}분 · 극한 실험] 프리셋 1~4번을 바꿔가며 극한 파라미터에서 그래프가 변하는 물리적 원인을 규명합니다."
                s3 = f"[3교시 {m3}분 · C99 코드] {unit['step3']}"

            day_items.append(
                StudyDayItem(
                    day_number=d_idx + 1,
                    date_str=cur_date.strftime("%Y-%m-%d"),
                    phase_badge=unit["phase"],
                    title=title,
                    target_module_id=unit["module_id"],
                    target_level=lvl,
                    recommended_preset_idx=int(unit["preset_idx"]),
                    hours=daily_hours,
                    step1_concept=s1,
                    step2_practice=s2,
                    step3_review_c=s3,
                    self_check_question=unit["check_q"],
                    user_notes="",
                    completed=False,
                )
            )

    return StudyPlanDocument(
        start_date_str=start_date.strftime("%Y-%m-%d"),
        total_days=total_days,
        daily_hours=daily_hours,
        focus_track=focus_track,
        days=day_items,
    )


def save_study_plan(plan: StudyPlanDocument, custom_path: Path | None = None) -> Path:
    """Save the user's StudyPlanDocument to JSON on disk."""
    target = custom_path if custom_path is not None else get_user_study_plan_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "start_date_str": plan.start_date_str,
        "total_days": plan.total_days,
        "daily_hours": plan.daily_hours,
        "focus_track": plan.focus_track,
        "days": [asdict(d) for d in plan.days],
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return target


def load_study_plan(custom_path: Path | None = None, raise_on_error: bool = False) -> StudyPlanDocument:
    """Load the user's saved StudyPlanDocument from disk, or generate a default 14-day plan starting today."""
    target = custom_path if custom_path is not None else get_user_study_plan_path()
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                raw = json.load(f)
            days = [StudyDayItem(**item) for item in raw.get("days", [])]
            if not days:
                raise ValueError("저장된 학습 계획표에 일차(days) 항목이 비어 있습니다.")
            return StudyPlanDocument(
                start_date_str=str(raw.get("start_date_str", date.today().strftime("%Y-%m-%d"))),
                total_days=int(raw.get("total_days", len(days))),
                daily_hours=float(raw.get("daily_hours", 2.0)),
                focus_track=str(raw.get("focus_track", "balanced")),
                days=days,
            )
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(f"저장된 학습 계획표 파일({target})을 불러오는 중 오류가 발생했습니다.\n\n상세 내용: {exc}") from exc
            raise
    return generate_study_plan(total_days=14, start_date=date.today(), daily_hours=2.0, focus_track="balanced")


def export_study_plan_markdown(plan: StudyPlanDocument) -> str:
    """Export the StudyPlanDocument into a clean GitHub-Flavored Markdown study schedule table."""
    done_cnt = sum(1 for d in plan.days if d.completed)
    total_h = sum(d.hours for d in plan.days)
    lines = [
        "# 단암시스템즈 항법기술연구소 SW팀 입사 전 맞춤 위성항법 학습 계획표",
        "",
        f"- **학습 시작일**: {plan.start_date_str} (총 **{len(plan.days)}일** 과정, 하루 **{plan.daily_hours:.1f}시간** / 총 **{total_h:.1f}시간**)",
        f"- **현재 달성률**: {done_cnt} / {len(plan.days)}일 완료 ({done_cnt / max(len(plan.days), 1) * 100:.0f}%)",
        "",
        "| 일차 | 날짜 | 단계 | 학습 주제 | 연동 모듈 | 추천 레벨 | 시간 | 완료 |",
        "| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for d in plan.days:
        chk = "[x]" if d.completed else "[ ]"
        lvl_str = {"lv1": "Lv.1 입문", "both": "Lv.1+2 통합", "lv2": "Lv.2 심화"}.get(d.target_level, d.target_level)
        lines.append(
            f"| **Day {d.day_number:02d}** | {d.date_str} | {d.phase_badge} | **{d.title}** | `{d.target_module_id}` | {lvl_str} | {d.hours:.1f}h | {chk} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("## 일차별 저녁 2시간(3교시) 상세 액션 플랜 및 셀프 체크")
    lines.append("")
    for d in plan.days:
        lines.append(f"### Day {d.day_number:02d} ({d.date_str}) — {d.title}")
        lines.append(f"- **1교시 (개념 · 비유)**: {d.step1_concept}")
        lines.append(f"- **2교시 (시뮬레이션 실습)**: {d.step2_practice}")
        lines.append(f"- **3교시 (정리 · C 포팅)**: {d.step3_review_c}")
        lines.append(f"- **오늘의 합격 셀프 체크**: *{d.self_check_question}*")
        if d.user_notes.strip():
            lines.append(f"- **내 학습 메모**: {d.user_notes.strip()}")
        lines.append("")

    return "\n".join(lines)
