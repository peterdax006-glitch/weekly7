"""Nupen's hands on the iPhone: taps and swipes through a Bluetooth mouse that the PC drives.

Hardware: an ESP32 board on USB (serial) runs firmware/phonetap_esp32 and shows up on the iPhone as a Bluetooth mouse. With
AssistiveTouch on, iOS draws a pointer; a left click is a tap. The pointer is RELATIVE (the mouse only says "move dx, dy"), so
PhoneTap keeps its own estimate of where the pointer is:

  * home(): push hard to the top-left corner (iOS stops the pointer at the edge) -> position known exactly = (0, 0);
  * move in many small steps of the same size (iOS pointer acceleration then behaves ~linearly) using `gain` = points per count;
  * calibrate(measure): measure() returns where the pointer really is (e.g. found in the mirrored screenshot from
    creator.phonescreen) -> gain fitted from a known move; re-home every `rehome_every` moves to cancel drift.

Coordinates are iPhone points (iPhone 13: 390 x 844). frame_to_points() maps a pixel in a mirrored screenshot to points.

Serial protocol (115200 baud, one line each way): "P" -> "OK P <paired 0|1>"; "M dx dy" (|d| <= 127) -> "OK"; "D" / "U" left
button down / up -> "OK"; "C <ms>" click -> "OK"; "W <dz>" scroll -> "OK". Anything else -> "ERR ...".
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional, Protocol

IPHONE_13 = (390, 844)


class Device(Protocol):
    def send(self, line: str) -> str: ...


class SerialDevice:
    """The real board. Needs pyserial (`pip install pyserial`); port like 'COM5' (Device Manager -> Ports)."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        import serial                                                                   # lazy: tests never need it
        self.s = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)                                                                  # boards reset when the port opens
        self.s.reset_input_buffer()

    def send(self, line: str) -> str:
        self.s.write((line + "\n").encode("ascii"))
        return self.s.readline().decode("ascii", "replace").strip()

    def close(self) -> None:
        self.s.close()


class FakeDevice:
    """A simulated phone pointer for tests: clamps at the screen edges, optional acceleration, records taps and drags."""

    def __init__(self, screen: tuple[int, int] = IPHONE_13, gain: float = 1.0, accel: float = 0.0, start: tuple[float, float] = (200, 400),
                 paired: bool = True):
        self.w, self.h = screen
        self.gain, self.accel = gain, accel
        self.x, self.y = start
        self.down = False
        self.taps: list[tuple[float, float]] = []
        self.drags: list[list[tuple[float, float]]] = []
        self.paired = paired
        self.lines: list[str] = []

    def _mv(self, c: int) -> float:
        return c * self.gain * (1 + self.accel * abs(c))

    def send(self, line: str) -> str:
        self.lines.append(line)
        p = line.split()
        if not p:
            return "ERR empty"
        if p[0] == "P":
            return f"OK P {int(self.paired)}"
        if not self.paired:
            return "ERR not paired"
        if p[0] == "M":
            dx, dy = int(p[1]), int(p[2])
            if abs(dx) > 127 or abs(dy) > 127:
                return "ERR range"
            self.x = min(max(self.x + self._mv(dx), 0), self.w - 1)
            self.y = min(max(self.y + self._mv(dy), 0), self.h - 1)
            if self.down:
                self.drags[-1].append((self.x, self.y))
            return "OK"
        if p[0] == "C":
            self.taps.append((self.x, self.y))
            return "OK"
        if p[0] == "D":
            self.down = True
            self.drags.append([(self.x, self.y)])
            return "OK"
        if p[0] == "U":
            self.down = False
            return "OK"
        if p[0] == "W":
            return "OK"
        return "ERR unknown"


class TapError(RuntimeError):
    pass


class PhoneTap:
    def __init__(self, dev: Device, screen: tuple[int, int] = IPHONE_13, gain: float = 1.0, step: int = 6, rehome_every: int = 8,
                 sleep: Callable[[float], None] = lambda s: None):
        self.dev, self.w, self.h = dev, screen[0], screen[1]
        self.gain, self.step, self.rehome_every = gain, max(1, min(step, 127)), rehome_every
        self.pos: Optional[tuple[float, float]] = None
        self.moves = 0
        self.sleep = sleep

    def _cmd(self, line: str) -> str:
        r = self.dev.send(line)
        if not r.startswith("OK"):
            raise TapError(f"{line!r} -> {r!r}")
        return r

    def paired(self) -> bool:
        r = self.dev.send("P")
        return r.startswith("OK P 1")

    def home(self) -> None:
        """Pin the pointer to the top-left corner: enough full-size moves to cross the whole screen even with gain < 1."""
        n = math.ceil(max(self.w, self.h) / (127 * max(self.gain, 0.05))) + 3
        for _ in range(n):
            self._cmd("M -127 -127")
        self.pos, self.moves = (0.0, 0.0), 0

    def _counts(self, d: float) -> list[int]:
        total = round(d / self.gain)
        sign = 1 if total >= 0 else -1
        out, left = [], abs(total)
        while left > 0:
            c = min(self.step, left)
            out.append(sign * c)
            left -= c
        return out

    def move_by(self, dx: float, dy: float) -> None:
        cx, cy = self._counts(dx), self._counts(dy)
        for i in range(max(len(cx), len(cy))):
            self._cmd(f"M {cx[i] if i < len(cx) else 0} {cy[i] if i < len(cy) else 0}")
            self.sleep(0.004)
        x0, y0 = self.pos or (0.0, 0.0)
        self.pos = (min(max(x0 + sum(cx) * self.gain, 0), self.w - 1), min(max(y0 + sum(cy) * self.gain, 0), self.h - 1))

    def move_to(self, x: float, y: float) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise TapError(f"({x}, {y}) is off the {self.w}x{self.h} screen")
        if self.pos is None or self.moves >= self.rehome_every:
            self.home()
        self.moves += 1
        self.move_by(x - self.pos[0], y - self.pos[1])

    def tap(self, x: float, y: float, hold_ms: int = 40) -> None:
        self.move_to(x, y)
        self._cmd(f"C {int(hold_ms)}")

    def swipe(self, x1: float, y1: float, x2: float, y2: float, steps: int = 12) -> None:
        self.move_to(x1, y1)
        self._cmd("D")
        try:
            for i in range(1, steps + 1):
                tx, ty = x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps
                self.move_by(tx - self.pos[0], ty - self.pos[1])
                self.sleep(0.01)
        finally:
            self._cmd("U")

    # ----------------------------------------------------------------------------------------- screenshots <-> points
    def frame_to_points(self, px: float, py: float, frame_w: int, frame_h: int) -> tuple[float, float]:
        return px * self.w / frame_w, py * self.h / frame_h

    def tap_frame(self, px: float, py: float, frame_w: int, frame_h: int) -> None:
        self.tap(*self.frame_to_points(px, py, frame_w, frame_h))

    def calibrate(self, measure: Callable[[], tuple[float, float]], counts: int = 120) -> float:
        """Home, move `counts` counts right and down in normal steps, measure the real pointer (points) -> new gain."""
        self.home()
        self.gain = 1.0
        for c in self._counts(counts):
            self._cmd(f"M {c} {c}")
        mx, my = measure()
        g = ((mx / counts) + (my / counts)) / 2
        if not 0.05 <= g <= 20:
            raise TapError(f"calibration gave gain {g:.3f}: pointer not found or not moving")
        self.gain = g
        self.pos = None
        return g
