from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PointCloudWindow:
    index: int
    label: str
    points: np.ndarray
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointCloudResult:
    method: str
    full_points: np.ndarray
    assets: List[str]
    windows: List[PointCloudWindow]
    summary_lines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

    def ensure_windows(self) -> List[PointCloudWindow]:
        if self.windows:
            return self.windows
        fallback = PointCloudWindow(index=0, label="Full Sample", points=self.full_points)
        return [fallback]
