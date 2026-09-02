"""
============================================================
audio.py

[역할]

MicArray A에서 DOA(Direction Of Arrival)에 필요한
멀티채널 오디오 데이터를 읽습니다.

중요
------------------------------------------------------------
이 파일에서는

- 500 Hz 검출
- 650 Hz 검출
- 900 Hz 검출
- 1050 Hz 검출
- X/Y 판정
- XYXY 검증
- LOCK

을 하지 않습니다.

그 부분은 전부 beacon1_lock_final.py 담당입니다.

이 파일은 LOCK 이후 방향 추정을 위한 MicArray A만 담당합니다.
============================================================
"""

import subprocess

import numpy as np

from config import (
    DOA_DEVICE,
    DOA_SAMPLE_RATE,
    DOA_TOTAL_CHANNELS,
    DEBUG,
)


class DOAAudioReader:
    """
    MicArray A의 오디오 스트림을 읽는 클래스.
    """

    def __init__(self):

        self.proc = None

        self.bytes_per_sample = 2  # S16_LE


    def start(self):
        """
        arecord를 실행하여 MicArray A 스트림을 시작합니다.
        """

        if self.proc is not None:
            return

        cmd = [
            "arecord",
            "-q",

            "-D",
            DOA_DEVICE,

            "-t",
            "raw",

            "-f",
            "S16_LE",

            "-r",
            str(DOA_SAMPLE_RATE),

            "-c",
            str(DOA_TOTAL_CHANNELS),
        ]

        if DEBUG:
            print(
                "[AUDIO] MicArray A 시작"
            )

            print(
                f"[AUDIO] device = {DOA_DEVICE}"
            )

            print(
                f"[AUDIO] sample rate = "
                f"{DOA_SAMPLE_RATE}"
            )

            print(
                f"[AUDIO] channels = "
                f"{DOA_TOTAL_CHANNELS}"
            )

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )


    def _read_exact(self, nbytes):
        """
        원하는 byte 수만큼 정확하게 읽습니다.
        """

        chunks = []

        received = 0

        while received < nbytes:

            chunk = self.proc.stdout.read(
                nbytes - received
            )

            if not chunk:
                break

            chunks.append(chunk)

            received += len(chunk)

        return b"".join(chunks)


    def read(self, duration_sec):
        """
        duration_sec 만큼 멀티채널 오디오를 읽습니다.

        반환:

        numpy.ndarray

        shape:
            [samples, channels]

        값:
            -1.0 ~ +1.0
        """

        if self.proc is None:
            self.start()

        samples = int(
            DOA_SAMPLE_RATE
            * duration_sec
        )

        nbytes = (
            samples
            * DOA_TOTAL_CHANNELS
            * self.bytes_per_sample
        )

        raw_bytes = self._read_exact(
            nbytes
        )

        if len(raw_bytes) != nbytes:

            raise RuntimeError(
                "MicArray A에서 필요한 만큼의 "
                "오디오를 읽지 못했습니다."
            )

        raw = np.frombuffer(
            raw_bytes,
            dtype="<i2",
        )

        raw = raw.reshape(
            -1,
            DOA_TOTAL_CHANNELS,
        )

        # int16 → -1.0 ~ 1.0
        data = (
            raw.astype(np.float64)
            / 32768.0
        )

        return data


    def stop(self):
        """
        arecord 종료
        """

        if self.proc is None:
            return

        if DEBUG:
            print(
                "[AUDIO] MicArray A 종료"
            )

        self.proc.terminate()

        try:

            self.proc.wait(
                timeout=2
            )

        except subprocess.TimeoutExpired:

            self.proc.kill()

        self.proc = None
