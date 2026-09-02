#!/usr/bin/env python3
"""BU03-Kit TWR distance reader.

The BU03-Kit responds to ``AT+DISTANCE`` with a line such as::

    distance: 0.019496
    OK

This module keeps the serial part independent from ROS2 so it can be tested
on the TurtleBot computer before it is connected to the navigation state
machine.  The returned distance is treated as metres, matching the BU03
factory AT firmware output used by this project.

Examples
--------
Collect ten samples and print median/std/MAD::

    python3 doa_angle/uwb_reader.py --port /dev/ttyUSB0 --count 10

Use the stable USB identity when available::

    python3 doa_angle/uwb_reader.py \
        --port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
        --count 20 --interval 0.2 --max-std 0.05

Run parser checks without hardware::

    python3 doa_angle/uwb_reader.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - exercised only on an unprepared host
    serial = None  # type: ignore[assignment]

    class SerialException(Exception):
        """Fallback exception used when pyserial is unavailable."""


DISTANCE_RE = re.compile(
    r"\bdistance\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

DEFAULT_STABLE_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
DEFAULT_FALLBACK_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_SEC = 1.5
DEFAULT_INTERVAL_SEC = 0.2
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_MIN_VALID_RATIO = 0.8
DEFAULT_MIN_VALID_FLOOR = 3
DEFAULT_MAX_STD_M = 0.05
DEFAULT_STARTUP_DELAY_SEC = 1.0
DEFAULT_HOLD_TIMEOUT_SEC = 3.0
DEFAULT_HOLD_EPSILON_M = 0.03


def choose_default_port() -> str:
    """Return the stable BU03 USB-TTL path when it exists."""

    if Path(DEFAULT_STABLE_PORT).exists():
        return DEFAULT_STABLE_PORT
    return DEFAULT_FALLBACK_PORT


def parse_distance(text: str) -> Optional[float]:
    """Extract one BU03 distance in metres from a response line/block."""

    match = DISTANCE_RE.search(text)
    if not match:
        return None

    value = float(match.group(1))
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def robust_mad(values: Sequence[float]) -> float:
    """Return the median absolute deviation in metres."""

    if not values:
        return float("nan")
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


@dataclass(frozen=True)
class DistanceSample:
    """One valid range response."""

    timestamp: str
    distance_m: float
    response: str
    latency_ms: float
    freshness: str = "FRESH"
    unchanged_sec: float = 0.0


class DistanceFreshnessTracker:
    """Classify whether successful responses contain a newly changed range.

    BU03's ``AT+DISTANCE`` response contains a distance and ``OK`` but does
    not expose a measurement sequence number or signal-quality field.  A
    repeated value is therefore reported as ``HOLD`` after a configurable
    timeout.  ``HOLD`` is a stale-value candidate, not an automatic failure:
    a stationary robot can legitimately observe the same distance.
    """

    def __init__(
        self,
        hold_timeout_sec: float = DEFAULT_HOLD_TIMEOUT_SEC,
        epsilon_m: float = DEFAULT_HOLD_EPSILON_M,
    ) -> None:
        if hold_timeout_sec <= 0.0:
            raise ValueError("hold_timeout_sec must be positive")
        if epsilon_m < 0.0:
            raise ValueError("epsilon_m must be non-negative")

        self.hold_timeout_sec = hold_timeout_sec
        self.epsilon_m = epsilon_m
        self._last_distance: Optional[float] = None
        self._last_change_time: Optional[float] = None

    def update(
        self,
        distance_m: float,
        now: Optional[float] = None,
    ) -> tuple[str, float]:
        """Return ``(freshness, unchanged_seconds)`` for one valid response."""

        observed_at = time.monotonic() if now is None else now
        if self._last_distance is None or self._last_change_time is None:
            self._last_distance = distance_m
            self._last_change_time = observed_at
            return "FRESH", 0.0

        if abs(distance_m - self._last_distance) > self.epsilon_m:
            self._last_distance = distance_m
            self._last_change_time = observed_at
            return "FRESH", 0.0

        unchanged_sec = max(0.0, observed_at - self._last_change_time)
        freshness = (
            "HOLD" if unchanged_sec >= self.hold_timeout_sec else "FRESH"
        )
        return freshness, unchanged_sec


@dataclass(frozen=True)
class DistanceSummary:
    """Statistics for a collection of valid BU03 measurements."""

    attempts: int
    valid_count: int
    median_m: Optional[float]
    mean_m: Optional[float]
    std_m: Optional[float]
    mad_m: Optional[float]
    robust_std_m: Optional[float]
    min_m: Optional[float]
    max_m: Optional[float]
    max_std_m: Optional[float]
    quality: str
    fresh_count: int = 0
    hold_count: int = 0


class BU03RangeReader:
    """Request and parse distance measurements from a BU03 serial port."""

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        response_timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        command: str = "AT+DISTANCE",
        line_ending: str = "crlf",
        startup_delay_sec: float = DEFAULT_STARTUP_DELAY_SEC,
    ) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial이 없습니다. 다음을 실행하세요: python3 -m pip install pyserial"
            )
        if response_timeout_sec <= 0.0:
            raise ValueError("response_timeout_sec must be positive")
        if startup_delay_sec < 0.0:
            raise ValueError("startup_delay_sec must be non-negative")

        endings = {"crlf": "\r\n", "cr": "\r", "lf": "\n"}
        if line_ending not in endings:
            raise ValueError(f"unsupported line ending: {line_ending}")

        self.port = port
        self.baudrate = baudrate
        self.response_timeout_sec = response_timeout_sec
        self.startup_delay_sec = startup_delay_sec
        self.command_bytes = (command + endings[line_ending]).encode("ascii")
        self._serial = None

    def __enter__(self) -> "BU03RangeReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=self.response_timeout_sec,
                exclusive=True,
            )
        except PermissionError as exc:
            raise PermissionError(
                f"{self.port}에 접근할 권한이 없습니다. "
                "사용자를 dialout 그룹에 추가한 뒤 다시 로그인하세요."
            ) from exc
        except SerialException as exc:
            raise RuntimeError(f"시리얼 포트를 열 수 없습니다: {self.port}: {exc}") from exc

        # A previous cat/serial session can leave old response bytes behind.
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        # Opening the CH340 port can toggle control lines and reset the BU03.
        # Give its firmware time to finish booting before the first AT command.
        if self.startup_delay_sec:
            time.sleep(self.startup_delay_sec)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def request_distance(self) -> tuple[float, str, float]:
        """Send one request and return ``(distance_m, response, latency_ms)``."""

        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("serial port is not open")

        self._serial.reset_input_buffer()
        started = time.monotonic()
        try:
            self._serial.write(self.command_bytes)
            self._serial.flush()
        except SerialException as exc:
            raise RuntimeError(f"BU03 명령 전송 실패: {exc}") from exc

        response_lines: list[str] = []
        distance_m: Optional[float] = None
        deadline = started + self.response_timeout_sec

        while time.monotonic() < deadline:
            raw_line = self._serial.readline()
            if not raw_line:
                continue

            line = raw_line.decode("ascii", errors="replace").strip()
            if not line:
                continue

            response_lines.append(line)
            parsed = parse_distance(line)
            if parsed is not None:
                distance_m = parsed

            # The factory firmware ends a successful distance response with OK.
            # ERR is returned when the command cannot be processed.
            if line.upper() == "ERR":
                response = "\\n".join(response_lines)
                raise RuntimeError(f"BU03가 ERR을 반환했습니다: {response}")
            if line.upper() == "OK" and distance_m is not None:
                break

        latency_ms = (time.monotonic() - started) * 1000.0
        response = "\\n".join(response_lines)
        if distance_m is None:
            if response:
                raise RuntimeError(f"거리값을 찾지 못했습니다. 응답: {response}")
            raise TimeoutError(
                f"{self.port}에서 {self.response_timeout_sec:.1f}초 동안 응답이 없습니다."
            )

        return distance_m, response, latency_ms


def summarize(
    samples: Sequence[DistanceSample],
    attempts: int,
    max_std_m: Optional[float],
    min_valid: int,
) -> DistanceSummary:
    values = [sample.distance_m for sample in samples]
    if not values:
        return DistanceSummary(
            attempts=attempts,
            valid_count=0,
            median_m=None,
            mean_m=None,
            std_m=None,
            mad_m=None,
            robust_std_m=None,
            min_m=None,
            max_m=None,
            max_std_m=max_std_m,
            quality="FAIL_NO_VALID_SAMPLE",
            fresh_count=0,
            hold_count=0,
        )

    median_m = statistics.median(values)
    std_m = statistics.pstdev(values) if len(values) >= 2 else 0.0
    mad_m = robust_mad(values)
    robust_std_m = 1.4826 * mad_m
    quality = "PASS"
    if len(values) < min_valid:
        quality = "FAIL_TOO_FEW_VALID"
    elif max_std_m is not None and std_m > max_std_m:
        quality = "FAIL_HIGH_STD"

    return DistanceSummary(
        attempts=attempts,
        valid_count=len(values),
        median_m=median_m,
        mean_m=statistics.mean(values),
        std_m=std_m,
        mad_m=mad_m,
        robust_std_m=robust_std_m,
        min_m=min(values),
        max_m=max(values),
        max_std_m=max_std_m,
        quality=quality,
        fresh_count=sum(sample.freshness == "FRESH" for sample in samples),
        hold_count=sum(sample.freshness == "HOLD" for sample in samples),
    )


def resolve_min_valid(sample_count: int, explicit_min_valid: Optional[int]) -> int:
    """Choose the minimum valid count for one measurement burst.

    A finite burst uses an 80% valid-response requirement.  Continuous mode
    has no fixed batch size, so it uses the safety floor of three samples for
    each summary produced by the caller.
    """

    if explicit_min_valid is not None:
        return explicit_min_valid
    if sample_count > 0:
        return max(
            DEFAULT_MIN_VALID_FLOOR,
            math.ceil(sample_count * DEFAULT_MIN_VALID_RATIO),
        )
    return DEFAULT_MIN_VALID_FLOOR


def write_csv_header(handle: TextIO) -> csv.DictWriter:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "timestamp",
            "distance_m",
            "latency_ms",
            "response",
            "freshness",
            "unchanged_sec",
        ],
    )
    if handle.tell() == 0:
        writer.writeheader()
    return writer


def collect_samples(args: argparse.Namespace) -> tuple[list[DistanceSample], int]:
    samples: list[DistanceSample] = []
    attempts = 0
    freshness_tracker = DistanceFreshnessTracker(
        hold_timeout_sec=args.hold_timeout,
        epsilon_m=args.hold_epsilon,
    )
    log_handle: Optional[TextIO] = None
    csv_writer: Optional[csv.DictWriter] = None

    try:
        if args.log:
            log_path = Path(args.log).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", newline="", encoding="utf-8")
            csv_writer = write_csv_header(log_handle)

        with BU03RangeReader(
            port=args.port,
            baudrate=args.baudrate,
            response_timeout_sec=args.timeout,
            command=args.command,
            line_ending=args.line_ending,
            startup_delay_sec=args.startup_delay,
        ) as reader:
            if not args.json:
                print(
                    f"[UWB] port={args.port} baudrate={args.baudrate} "
                    f"count={'continuous' if args.count == 0 else args.count}"
                )

            while args.count == 0 or attempts < args.count:
                attempts += 1
                try:
                    distance_m, response, latency_ms = reader.request_distance()
                except (RuntimeError, TimeoutError) as exc:
                    print(f"[UWB] {attempts}: 측정 실패: {exc}", file=sys.stderr)
                else:
                    freshness, unchanged_sec = freshness_tracker.update(distance_m)
                    sample = DistanceSample(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        distance_m=distance_m,
                        response=response,
                        latency_ms=latency_ms,
                        freshness=freshness,
                        unchanged_sec=unchanged_sec,
                    )
                    samples.append(sample)
                    if not args.json:
                        print(
                            f"[UWB] {attempts}: distance={distance_m:.6f} m "
                            f"latency={latency_ms:.1f} ms "
                            f"freshness={freshness} "
                            f"unchanged={unchanged_sec:.1f} s"
                        )
                    if csv_writer is not None and log_handle is not None:
                        csv_writer.writerow(asdict(sample))
                        log_handle.flush()

                if args.count == 0 or attempts < args.count:
                    time.sleep(args.interval)
    finally:
        if log_handle is not None:
            log_handle.close()

    return samples, attempts


def print_summary(summary: DistanceSummary, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return

    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.6f} m"

    print("[UWB] summary")
    print(f"  valid={summary.valid_count}/{summary.attempts}")
    print(f"  median={fmt(summary.median_m)}")
    print(f"  mean={fmt(summary.mean_m)}")
    print(f"  std={fmt(summary.std_m)}")
    print(f"  MAD={fmt(summary.mad_m)}")
    print(f"  robust_std={fmt(summary.robust_std_m)}")
    print(f"  min={fmt(summary.min_m)} max={fmt(summary.max_m)}")
    print(
        f"  freshness={summary.fresh_count} "
        f"hold={summary.hold_count}"
    )
    print(f"  quality={summary.quality}")


def run_self_test() -> int:
    cases = {
        "distance: 0.019496": 0.019496,
        "distance:0.340000\\nOK": 0.340000,
        "DISTANCE: 1.25e+00": 1.25,
    }
    for text, expected in cases.items():
        actual = parse_distance(text)
        if actual is None or not math.isclose(actual, expected):
            print(f"FAIL: {text!r} -> {actual!r}, expected {expected!r}")
            return 1

    if parse_distance("ERR") is not None:
        print("FAIL: ERR must not produce a distance")
        return 1

    values = [0.99, 1.00, 1.01, 1.00]
    samples = [
        DistanceSample("test", value, "distance", 1.0) for value in values
    ]
    result = summarize(samples, attempts=4, max_std_m=0.05, min_valid=3)
    if result.quality != "PASS" or not math.isclose(result.median_m or 0.0, 1.0):
        print(f"FAIL: summary={result}")
        return 1

    if resolve_min_valid(10, None) != 8:
        print("FAIL: finite burst minimum-valid policy")
        return 1

    tracker = DistanceFreshnessTracker(hold_timeout_sec=2.0, epsilon_m=0.03)
    if tracker.update(1.0, now=0.0) != ("FRESH", 0.0):
        print("FAIL: first freshness sample")
        return 1
    if tracker.update(1.02, now=1.0)[0] != "FRESH":
        print("FAIL: short repeated-value hold")
        return 1
    freshness, unchanged_sec = tracker.update(1.02, now=2.1)
    if freshness != "HOLD" or not math.isclose(unchanged_sec, 2.1):
        print("FAIL: stale-value hold detection")
        return 1
    if tracker.update(1.10, now=2.2) != ("FRESH", 0.0):
        print("FAIL: changed value must reset hold state")
        return 1

    noisy = [
        DistanceSample("test", value, "distance", 1.0)
        for value in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]
    noisy_result = summarize(noisy, attempts=5, max_std_m=0.05, min_valid=4)
    if noisy_result.quality != "FAIL_HIGH_STD":
        print(f"FAIL: noisy summary={noisy_result}")
        return 1

    print("self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BU03-Kit에 AT+DISTANCE를 보내 거리값을 자동 수집합니다."
    )
    parser.add_argument(
        "--port",
        default=choose_default_port(),
        help="BU03 serial port (default: %(default)s)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Serial baudrate (default: {DEFAULT_BAUDRATE})",
    )
    parser.add_argument(
        "--command",
        default="AT+DISTANCE",
        help="AT command without line ending (default: %(default)s)",
    )
    parser.add_argument(
        "--line-ending",
        choices=("crlf", "cr", "lf"),
        default="crlf",
        help="Command line ending (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Response timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=DEFAULT_STARTUP_DELAY_SEC,
        help=(
            "Delay after opening the port, for BU03 boot/reset "
            f"(default: {DEFAULT_STARTUP_DELAY_SEC})"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"Delay between requests in seconds (default: {DEFAULT_INTERVAL_SEC})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of requests; 0 means continuous (default: 10)",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=None,
        help=(
            "Override minimum valid samples; default is 80%% of a finite "
            f"burst, floor {DEFAULT_MIN_VALID_FLOOR}"
        ),
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=DEFAULT_MAX_STD_M,
        help=(
            "Maximum standard deviation in metres "
            f"(default: {DEFAULT_MAX_STD_M})"
        ),
    )
    parser.add_argument(
        "--hold-timeout",
        type=float,
        default=DEFAULT_HOLD_TIMEOUT_SEC,
        help=(
            "Mark an unchanged successful value as HOLD after this many "
            f"seconds (default: {DEFAULT_HOLD_TIMEOUT_SEC})"
        ),
    )
    parser.add_argument(
        "--hold-epsilon",
        type=float,
        default=DEFAULT_HOLD_EPSILON_M,
        help=(
            "Distance change in metres required to reset HOLD timing "
            f"(default: {DEFAULT_HOLD_EPSILON_M})"
        ),
    )
    parser.add_argument(
        "--log",
        help="Optional CSV path for raw valid measurements",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the final summary as JSON",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser/statistics tests without opening a serial port",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.count < 0:
        parser.error("--count must be >= 0")
    if args.interval < 0.0:
        parser.error("--interval must be >= 0")
    if args.startup_delay < 0.0:
        parser.error("--startup-delay must be >= 0")
    if args.min_valid is not None and args.min_valid < 1:
        parser.error("--min-valid must be >= 1")
    if args.max_std is not None and args.max_std < 0.0:
        parser.error("--max-std must be >= 0")
    if args.hold_timeout <= 0.0:
        parser.error("--hold-timeout must be > 0")
    if args.hold_epsilon < 0.0:
        parser.error("--hold-epsilon must be >= 0")

    min_valid = resolve_min_valid(args.count, args.min_valid)

    try:
        samples, attempts = collect_samples(args)
    except KeyboardInterrupt:
        print("[UWB] 사용자 중단", file=sys.stderr)
        return 130
    except (PermissionError, RuntimeError, SerialException) as exc:
        print(f"[UWB] 시작 실패: {exc}", file=sys.stderr)
        return 2

    summary = summarize(
        samples,
        attempts=attempts,
        max_std_m=args.max_std,
        min_valid=min_valid,
    )
    print_summary(summary, as_json=args.json)
    return 0 if summary.quality == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
