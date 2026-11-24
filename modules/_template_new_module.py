"""
================================================================================
[확장 템플릿] 새 학습/시뮬레이션 모듈 추가 템플릿 (`modules/_template_new_module.py`)
================================================================================
새 모듈(예: M06. RTK 반송파 위상 미지정수(LAMBDA), M07. 스푸핑(Spoofing) 탐지, M08. STM32 UART 실시간 연동)을
추가하는 방법:

1. 이 파일을 `modules/m06_my_new_topic.py`로 복사합니다.
2. `core/` 폴더에 UI 의존성이 없는 순수 NumPy/SciPy 수학 함수를 작성합니다.
3. 아래 `TemplateNewLabModule` 클래스의 메타데이터와 7개 표준 메서드를 구현합니다.
4. `ui/registry.py`의 `register_all_modules()`에 아래 단 1줄을 추가하면 사이드바에 자동 등록됩니다!
   >>> ModuleRegistry.register(TemplateNewLabModule)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ui.base_module import BaseLabModule
from ui.formula_widget import FormulaSpec, SymbolSpec


class TemplateNewLabModule(BaseLabModule):
    """Example template for extending GNSS SimLab with a new module (e.g. M06 RTK / Anti-Spoofing)."""

    module_id = "m06_template_demo"
    module_title = "M06. [확장 템플릿] RTK 반송파 위상 & 스푸핑 탐지"
    module_subtitle = "BaseLabModule 상속 및 ui/registry.py 1줄 등록을 통한 플러그인 확장 예제"
    module_badge = "MOD 06"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.run_simulation()

    def build_parameter_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("1. 예제 파라미터 제어")
        form = QFormLayout(group)

        self.slider_demo = QSlider(Qt.Orientation.Horizontal)
        self.slider_demo.setRange(1, 50)
        self.slider_demo.setValue(15)
        self.lbl_demo = QLabel("15")
        self.slider_demo.valueChanged.connect(self._on_demo_changed)

        form.addRow("데모 파라미터:", self.slider_demo)
        form.addRow("현재 값:", self.lbl_demo)
        layout.addWidget(group)
        return container

    def _on_demo_changed(self, val: int) -> None:
        self.lbl_demo.setText(str(val))
        self.run_simulation()

    def build_simulation_Plot_widget(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        self.fig = Figure(figsize=(8.0, 5.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)
        return container

    def apply_preset(self, preset_data: Dict[str, Any]) -> None:
        self.slider_demo.blockSignals(True)
        val = int(preset_data.get("demo_val", 15))
        self.slider_demo.setValue(val)
        self.lbl_demo.setText(str(val))
        self.slider_demo.blockSignals(False)

    def run_simulation(self) -> None:
        freq = float(self.slider_demo.value()) * 0.1
        t = np.linspace(0.0, 10.0, 300)
        y = np.sin(2.0 * np.pi * freq * t) * np.exp(-0.15 * t)

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(t, y, color="#1E3A2F", linewidth=2.0, label=f"Damped Oscillation (f={freq:.1f} Hz)")
        ax.set_title("새 모듈 확장 템플릿 그래프", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        self.finalize_2d_figure(self.fig)
        self.canvas.draw_idle()

        if hasattr(self, "fig_3d"):
            self.fig_3d.clear()
            ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection="3d")
            z = np.cos(2.0 * np.pi * freq * t) * np.exp(-0.15 * t)
            ax_3d.plot(t, y, z, color="#1E3A2F", linewidth=2.0)
            ax_3d.set_title("[3D] 새 모듈 3D 궤적 템플릿", fontsize=10.5, fontweight="bold", pad=10)
            ax_3d.set_xlabel("Time [s]")
            ax_3d.set_ylabel("Sin")
            ax_3d.set_zlabel("Cos")
            self.fig_3d.tight_layout(pad=1.4)
            self.canvas_3d.draw_idle()

    def get_theory_html(self) -> str:
        return """
        <h3>새 모듈 이론 해설 작성 영역</h3>
        <p>
            본문 내에 <a href="glossary://rtk_carrier_phase" style="color:#2563eb; font-weight:700;">RTK 반송파 위상</a> 또는
            <a href="glossary://spoofing" style="color:#2563eb; font-weight:700;">스푸핑(Spoofing)</a>과 같이
            <code>glossary://&lt;term_id&gt;</code> 링크를 넣으면 클릭 시 즉시 용어 해설 팝업이 표시됩니다.
        </p>
        """

    def get_formula_specs(self) -> List[FormulaSpec]:
        return [
            FormulaSpec(
                title="수식 1. RTK 반송파 위상 이중차분(Double-Difference) 관측 방정식",
                latex_expr=r"\nabla\Delta \Phi_{AB}^{jk} = \nabla\Delta r_{AB}^{jk} + \lambda_{L1} \nabla\Delta N_{AB}^{jk} + \varepsilon_{\Phi}",
                intuitive_summary="기준국(A)-이동국(B)과 두 위성(j, k) 간의 이중차분을 통해 위성·수신기 시계 오차와 대기 오차를 소거하고 정수 미지정수 N만 남깁니다.",
                symbols=[
                    SymbolSpec(r"\nabla\Delta \Phi_{AB}^{jk}", "이중차분 반송파 위상", "m", "시계 및 대기 오차가 소거된 정밀 반송파 거리 측정치", "rtk_carrier_phase"),
                    SymbolSpec(r"\lambda_{L1}", "GPS L1 반송파 파장", "m", "c / 1575.42 MHz ≈ 0.1903 m (19.03 cm)", "rtk_carrier_phase"),
                    SymbolSpec(r"\nabla\Delta N_{AB}^{jk}", "이중차분 정수 미지정수", "cycles", "LAMBDA 알고리즘으로 정수 확정(Fix)해야 하는 파장 수", "rtk_carrier_phase"),
                ],
            )
        ]

    def get_observation_guide_html(self) -> str:
        return """
        <b style="color:#1E3A2F;">실습 관찰 가이드</b>
        <p>좌측 슬라이더를 움직여 실시간 그래프 변화를 관찰하세요.</p>
        """

    def get_embedded_c_porting_html(self) -> str:
        return """
        <h3 style="color:#1E3A2F;">임베디드 C 포팅 노트</h3>
        <pre style="background-color:#0f172a; color:#e2e8f0; padding:12px; border-radius:6px;">
/* 새 모듈의 C99 / STM32 / DSP 포팅 예제 코드를 이곳에 작성합니다. */</pre>
        """
