"""
Central Module Registry for GNSS SimLab.
Adding a new learning module requires only placing the file in `modules/`
and adding 1 line `ModuleRegistry.register(NewModuleClass)` in `register_all_modules()`.
"""

from __future__ import annotations

from typing import List, Type

from ui.base_module import BaseLabModule


class ModuleRegistry:
    """Central registry storing ordered BaseLabModule subclasses."""

    _modules: List[Type[BaseLabModule]] = []

    @classmethod
    def register(cls, module_cls: Type[BaseLabModule]) -> None:
        if module_cls not in cls._modules:
            cls._modules.append(module_cls)

    @classmethod
    def get_registered_modules(cls) -> List[Type[BaseLabModule]]:
        if not cls._modules:
            register_all_modules()
        return list(cls._modules)


def register_all_modules() -> None:
    """
    Register all learning modules in sidebar display order.
    To add a future module (e.g. RTK, Spoofing, STM32 Serial Bridge), add 1 line here.
    """
    from modules.m01_coords_time import Module01CoordsTime
    from modules.m02_signal_acq_track import Module02SignalAcqTrack
    from modules.m03_pvt_dop import Module03PvtDop
    from modules.m04_antijam_crpa import Module04AntiJamCrpa
    from modules.m05_gnss_ins_ekf import Module05GnssInsEkf
    from modules.m06_egi_raim_spoof import Module06EgiRaimSpoof

    ModuleRegistry.register(Module01CoordsTime)
    ModuleRegistry.register(Module02SignalAcqTrack)
    ModuleRegistry.register(Module03PvtDop)
    ModuleRegistry.register(Module04AntiJamCrpa)
    ModuleRegistry.register(Module05GnssInsEkf)
    ModuleRegistry.register(Module06EgiRaimSpoof)
