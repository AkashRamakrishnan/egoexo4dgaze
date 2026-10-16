"""Minimal LZ-string URI decoder for Ego-Exo4D Relations masks."""

from __future__ import annotations

import math

KEYSTRURISAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-$"
_BASE_REVERSE: dict[str, dict[str, int]] = {}


class _Object:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get_base_value(alphabet: str, character: str) -> int:
    if alphabet not in _BASE_REVERSE:
        _BASE_REVERSE[alphabet] = {alphabet[i]: i for i in range(len(alphabet))}
    return _BASE_REVERSE[alphabet][character]


def _decompress(length: int, reset_value: int, get_next_value):
    dictionary: dict[int, str] = {}
    enlarge_in = 4
    dict_size = 4
    num_bits = 3
    entry = ""
    result: list[str] = []

    data = _Object(val=get_next_value(0), position=reset_value, index=1)

    for i in range(3):
        dictionary[i] = chr(i)

    bits = 0
    maxpower = math.pow(2, 2)
    power = 1

    while power != maxpower:
        resb = data.val & data.position
        data.position >>= 1
        if data.position == 0:
            data.position = reset_value
            data.val = get_next_value(data.index)
            data.index += 1
        bits |= int(power) if resb > 0 else 0
        power <<= 1

    next_val = bits
    if next_val == 0:
        bits = 0
        maxpower = math.pow(2, 8)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= int(power) if resb > 0 else 0
            power <<= 1
        c = chr(bits)
    elif next_val == 1:
        bits = 0
        maxpower = math.pow(2, 16)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= int(power) if resb > 0 else 0
            power <<= 1
        c = chr(bits)
    elif next_val == 2:
        return ""

    dictionary[3] = c
    w = c
    result.append(c)

    while True:
        if data.index > length:
            return ""

        bits = 0
        maxpower = math.pow(2, num_bits)
        power = 1
        while power != maxpower:
            resb = data.val & data.position
            data.position >>= 1
            if data.position == 0:
                data.position = reset_value
                data.val = get_next_value(data.index)
                data.index += 1
            bits |= int(power) if resb > 0 else 0
            power <<= 1

        c = bits
        if c == 0:
            bits = 0
            maxpower = math.pow(2, 8)
            power = 1
            while power != maxpower:
                resb = data.val & data.position
                data.position >>= 1
                if data.position == 0:
                    data.position = reset_value
                    data.val = get_next_value(data.index)
                    data.index += 1
                bits |= int(power) if resb > 0 else 0
                power <<= 1
            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c = dict_size - 1
            enlarge_in -= 1
        elif c == 1:
            bits = 0
            maxpower = math.pow(2, 16)
            power = 1
            while power != maxpower:
                resb = data.val & data.position
                data.position >>= 1
                if data.position == 0:
                    data.position = reset_value
                    data.val = get_next_value(data.index)
                    data.index += 1
                bits |= int(power) if resb > 0 else 0
                power <<= 1
            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c = dict_size - 1
            enlarge_in -= 1
        elif c == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = int(math.pow(2, num_bits))
            num_bits += 1

        if c in dictionary:
            entry = dictionary[c]
        elif c == dict_size:
            entry = w + w[0]
        else:
            return None

        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        enlarge_in -= 1
        w = entry

        if enlarge_in == 0:
            enlarge_in = int(math.pow(2, num_bits))
            num_bits += 1


def decompress_from_encoded_uri(compressed: str | None) -> str | None:
    if compressed is None:
        return ""
    if compressed == "":
        return None
    compressed = compressed.replace(" ", "+")
    return _decompress(
        len(compressed),
        32,
        lambda index: _get_base_value(KEYSTRURISAFE, compressed[index]),
    )
