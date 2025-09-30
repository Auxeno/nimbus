"""
Input module for polling and parsing gamepad state.

`evdev` is used for Linux
`hid` for MacOS
Windows is untested
"""

import platform
import threading
import time

IS_LINUX = platform.system() == "Linux"

if IS_LINUX:
    from evdev import InputDevice, ecodes, list_devices  # type: ignore
else:
    import hid

# Xbox controller vendor and product ID
VENDOR_ID = 0x045E
PRODUCT_ID = 0x0B12


def parse_xbox_bytes(report: list[int]) -> dict:
    """Convert a Xbox Controller HID bytes into a clean dictionary."""

    def axis(lo, hi) -> float:
        val = int.from_bytes([lo, hi], byteorder="little", signed=True)
        return round(val / 32767, 3)

    byte4, byte5 = report[4], report[5]

    return {
        # Buttons
        "start": bool(byte4 & 0x04),
        "select": bool(byte4 & 0x08),
        "a": bool(byte4 & 0x10),
        "b": bool(byte4 & 0x20),
        "x": bool(byte4 & 0x40),
        "y": bool(byte4 & 0x80),
        "dpad_u": bool(byte5 & 0x01),
        "dpad_d": bool(byte5 & 0x02),
        "dpad_l": bool(byte5 & 0x04),
        "dpad_r": bool(byte5 & 0x08),
        "lb": bool(byte5 & 0x10),
        "rb": bool(byte5 & 0x20),
        "l3": bool(byte5 & 0x40),
        "r3": bool(byte5 & 0x80),
        # Triggers (report[6,8]), normalised [0, +1]
        "l_trigger": round(report[6] / 255, 3),
        "r_trigger": round(report[8] / 255, 3),
        # Sticks (report[10–17]), normalised [−1, +1]
        "l_stick": (axis(report[10], report[11]), axis(report[12], report[13])),
        "r_stick": (axis(report[14], report[15]), axis(report[16], report[17])),
    }


def parse_evdev_event(event, state):
    """Update state dictionary with evdev event data."""
    axis_map = {
        ecodes.ABS_X: ("l_stick", 0),
        ecodes.ABS_Y: ("l_stick", 1),
        ecodes.ABS_RX: ("r_stick", 0),
        ecodes.ABS_RY: ("r_stick", 1),
        ecodes.ABS_Z: "l_trigger",
        ecodes.ABS_RZ: "r_trigger",
        ecodes.ABS_HAT0X: ("dpad_l", "dpad_r"),
        ecodes.ABS_HAT0Y: ("dpad_u", "dpad_d"),
    }

    button_map = {
        ecodes.BTN_SOUTH: "a",
        ecodes.BTN_EAST: "b",
        ecodes.BTN_NORTH: "x",
        ecodes.BTN_WEST: "y",
        ecodes.BTN_START: "start",
        ecodes.BTN_SELECT: "select",
        ecodes.BTN_TL: "lb",
        ecodes.BTN_TR: "rb",
        ecodes.BTN_THUMBL: "l3",
        ecodes.BTN_THUMBR: "r3",
    }

    if event.type == ecodes.EV_KEY:
        name = button_map.get(event.code)
        if name:
            state[name] = bool(event.value)

    elif event.type == ecodes.EV_ABS:
        code = event.code
        val = event.value

        if code in (ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX, ecodes.ABS_RY):
            stick, idx = axis_map[code]
            norm = round(val / 32768, 3)
            if code in (ecodes.ABS_Y, ecodes.ABS_RY):
                norm = -norm  # Invert vertical axes
            state[stick][idx] = norm

        elif code in (ecodes.ABS_Z, ecodes.ABS_RZ):
            norm = round(val / 1023, 3)
            state[axis_map[code]] = norm

        elif code == ecodes.ABS_HAT0X:
            state["dpad_l"] = val == -1
            state["dpad_r"] = val == 1

        elif code == ecodes.ABS_HAT0Y:
            state["dpad_u"] = val == -1
            state["dpad_d"] = val == 1


class Gamepad:
    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()
        self._running = True

        if IS_LINUX:
            # Locate the Xbox controller device
            devices = [InputDevice(path) for path in list_devices()]
            for dev in devices:
                if "Xbox" in dev.name:
                    self.device = dev
                    break
            else:
                raise RuntimeError("No Xbox controller found on Linux.")

            self._state = {
                "start": False,
                "select": False,
                "a": False,
                "b": False,
                "x": False,
                "y": False,
                "dpad_u": False,
                "dpad_d": False,
                "dpad_l": False,
                "dpad_r": False,
                "lb": False,
                "rb": False,
                "l3": False,
                "r3": False,
                "l_trigger": 0.0,
                "r_trigger": 0.0,
                "l_stick": [0.0, 0.0],
                "r_stick": [0.0, 0.0],
            }
        else:
            self.device = hid.device()  # type: ignore
            self.device.open(VENDOR_ID, PRODUCT_ID)
            self.device.set_nonblocking(True)

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        if IS_LINUX:
            for event in self.device.read_loop():
                with self._lock:
                    parse_evdev_event(event, self._state)
                if not self._running:
                    break
        else:
            while self._running:
                report = self.device.read(64)
                if report:
                    parsed = parse_xbox_bytes(report)
                    with self._lock:
                        self._state = parsed
                time.sleep(0.005)

    def poll(self) -> dict:
        with self._lock:
            return dict(self._state)

    def close(self):
        self._running = False
        self._thread.join()
        if not IS_LINUX:
            self.device.close()


def print_devices():
    """
    Use this if not using an Xbox controller over USB.
    Controllers and their product IDs can be labelled.
    Users may need to manually write the byte parsing logic for their own
    controllers.
    """
    if IS_LINUX:
        from evdev import InputDevice, list_devices  # type: ignore

        for dev in [InputDevice(path) for path in list_devices()]:
            print(f"{dev.path}: {dev.name}")
    else:
        for device in hid.enumerate():
            print(
                f"0x{device['vendor_id']:04x}:0x{device['product_id']:04x} {device['product_string']}"
            )
