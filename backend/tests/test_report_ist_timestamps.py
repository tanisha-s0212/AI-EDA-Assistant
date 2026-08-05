"""Report timestamps should render in Indian Standard Time (Asia/Kolkata)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import format_ist_date, format_ist_display, now_ist  # noqa: E402


def test_format_ist_display_converts_utc_iso_to_ist():
    # 08:40 UTC on 4 Aug 2026 → 14:10 IST same day
    assert format_ist_display('2026-08-04T08:40:04+00:00') == '04 Aug 2026, 02:10 PM IST'


def test_format_ist_display_treats_naive_iso_as_utc():
    # Cleaning logs use utcnow().isoformat() without offset.
    assert format_ist_display('2026-08-04T08:40:04.410360') == '04 Aug 2026, 02:10 PM IST'


def test_format_ist_display_aware_datetime():
    stamp = datetime(2026, 8, 4, 8, 40, 4, tzinfo=timezone.utc)
    assert format_ist_display(stamp) == '04 Aug 2026, 02:10 PM IST'


def test_format_ist_date_from_utc():
    assert format_ist_date('2026-08-04T20:30:00+00:00') == '2026-08-05'


def test_now_ist_is_asia_kolkata():
    stamp = now_ist()
    assert str(stamp.tzinfo) in {'Asia/Kolkata', 'IST'}
    assert stamp.tzname() in {'IST', 'Asia/Kolkata'} or stamp.utcoffset().total_seconds() == 19800
