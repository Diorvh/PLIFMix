from __future__ import annotations

import re


def select_frame_indices(
    frame_count: int,
    *,
    time_start: float,
    time_end: float,
    sample_frequency: float,
    use_frame_segments: bool,
    frame_segments: str,
) -> list[int]:
    if use_frame_segments:
        if not frame_segments.strip():
            raise ValueError("Non-contiguous interval mode is selected, but no intervals were entered.")
        selected: list[int] = []
        for segment in re.split(r"[,，;；]+", frame_segments):
            try:
                start_text, end_text = segment.strip().split(":", 1)
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid interval '{segment}'. Use start:end,start:end.") from exc
            if start < 0 or end <= start or end > frame_count:
                raise ValueError(f"Interval {start}:{end} is outside the available frame range 0:{frame_count}.")
            selected.extend(range(start, end))
        return selected
    if sample_frequency <= 0:
        raise ValueError("Sample frequency must be greater than zero.")
    start = int(time_start * sample_frequency)
    end = int(time_end * sample_frequency)
    if start < 0 or end <= start or end > frame_count:
        raise ValueError(f"Time range selects frames {start}:{end}, outside the available range 0:{frame_count}.")
    return list(range(start, end))
