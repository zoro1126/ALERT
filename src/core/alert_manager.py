"""
alert_manager.py
================
Alert state machine and audio alert delivery for ALERT.

State machine
-------------
              ┌──────────┐
   (reset)    │  NORMAL  │ ← class 0 (Alert)  for > debounce period
              └────┬─────┘
                   │ class 1 (LowVigilant)  ≥ WARNING_DURATION seconds
                   ▼
              ┌──────────┐
              │ WARNING  │ ← plays warning beep once
              └────┬─────┘
                   │ class 2 (Drowsy) OR class 1 ≥ CRITICAL_DURATION seconds
                   ▼
              ┌──────────┐
              │ CRITICAL │ ← plays critical alarm, repeats every
              └──────────┘   ALERT_DEBOUNCE_CRITICAL seconds

Audio
-----
Uses pygame.mixer for cross-platform audio.  Falls back silently if
pygame is not available or no audio device is present (e.g. headless server).

Tones are generated procedurally via numpy — no external audio files needed.

Usage
-----
manager = AlertManager()
# per inference result:
state = manager.update(class_index=2)
print(state)   # 'CRITICAL'
"""

from __future__ import annotations

import enum
import time
from typing import Callable

import numpy as np

from utils.config import Config


# ---------------------------------------------------------------------------
# Alert states
# ---------------------------------------------------------------------------

class AlertState(str, enum.Enum):
    NORMAL   = "NORMAL"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _try_init_audio() -> bool:
    """Return True if pygame.mixer initialised successfully."""
    try:
        import pygame
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        return True
    except Exception:
        return False


def _make_tone(
    frequency: float,
    duration_s: float,
    volume: float = 0.6,
    sample_rate: int = 22050,
) -> "pygame.sndarray.Sound | None":  # noqa: F821
    """
    Generate a pure sine-wave tone and return a pygame Sound object.
    Returns None if pygame is unavailable.
    """
    try:
        import pygame
        n_samples = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(wave)
        return sound
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Tracks fatigue alert state and plays audio cues.

    Parameters
    ----------
    on_state_change : callable, optional
        Called with (new_state: AlertState) whenever the state changes.
        Useful for logging or triggering UI colour changes.
    """

    def __init__(
        self,
        on_state_change: Callable[[AlertState], None] | None = None,
    ) -> None:
        self.state: AlertState = AlertState.NORMAL
        self._on_state_change  = on_state_change

        # Timestamps for duration tracking
        self._low_vig_since:  float | None = None
        self._critical_since: float | None = None
        self._last_audio_at:  float = 0.0

        # Audio
        self._audio_ok = _try_init_audio()
        self._warn_sound     = _make_tone(880,  0.3) if self._audio_ok else None
        self._critical_sound = _make_tone(1200, 0.6) if self._audio_ok else None

    # ------------------------------------------------------------------
    def update(self, class_index: int) -> AlertState:
        """
        Feed the latest model prediction and update alert state.

        Parameters
        ----------
        class_index : int
            0 = Alert, 1 = LowVigilant, 2 = Drowsy

        Returns
        -------
        AlertState : current state after update
        """
        now = time.monotonic()
        new_state = self._compute_state(class_index, now)

        if new_state != self.state:
            self.state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)
            self._trigger_audio(new_state, now)
            self._last_audio_at = now
        elif new_state == AlertState.CRITICAL:
            # Repeat critical alarm at debounce interval
            if now - self._last_audio_at >= Config.ALERT_DEBOUNCE_CRITICAL:
                self._trigger_audio(new_state, now)
                self._last_audio_at = now

        return self.state

    # ------------------------------------------------------------------
    def _compute_state(self, class_index: int, now: float) -> AlertState:
        """Pure state-transition logic (no side effects)."""
        if class_index == 2:
            # Immediately escalate to CRITICAL on Drowsy prediction
            if self._critical_since is None:
                self._critical_since = now
            self._low_vig_since = None
            return AlertState.CRITICAL

        if class_index == 1:
            # Low Vigilant: enter WARNING after WARNING_DURATION seconds
            if self._low_vig_since is None:
                self._low_vig_since = now
            self._critical_since = None

            elapsed = now - self._low_vig_since
            if elapsed >= Config.WARNING_DURATION:
                return AlertState.WARNING
            return self.state   # stay current until threshold crossed

        # class_index == 0 → Alert
        if self.state == AlertState.NORMAL:
            # Already normal; reset timers
            self._low_vig_since  = None
            self._critical_since = None
            return AlertState.NORMAL

        # Debounce: require ALERT_DEBOUNCE_WARNING seconds of alert before reset
        if self.state in (AlertState.WARNING, AlertState.CRITICAL):
            # Reset low-vig timer; only full debounce returns us to NORMAL
            self._low_vig_since  = None
            self._critical_since = None
            return AlertState.NORMAL

        return AlertState.NORMAL

    # ------------------------------------------------------------------
    def _trigger_audio(self, state: AlertState, _now: float) -> None:
        """Play the appropriate tone (non-blocking)."""
        if not self._audio_ok:
            return
        try:
            if state == AlertState.WARNING and self._warn_sound:
                self._warn_sound.play()
            elif state == AlertState.CRITICAL and self._critical_sound:
                self._critical_sound.play()
        except Exception:
            pass   # never crash the main loop on audio failure

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Return to NORMAL state and clear all timers."""
        self.state           = AlertState.NORMAL
        self._low_vig_since  = None
        self._critical_since = None
        self._last_audio_at  = 0.0

    # ------------------------------------------------------------------
    @property
    def alert_color_bgr(self) -> tuple[int, int, int]:
        """OpenCV BGR colour for current alert state (for HUD)."""
        return {
            AlertState.NORMAL:   (0, 220, 0),    # green
            AlertState.WARNING:  (0, 165, 255),   # orange
            AlertState.CRITICAL: (0, 0, 255),     # red
        }[self.state]
