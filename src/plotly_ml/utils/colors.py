import re


def _int_if_float(x):
    """Convert a float between 0 and 1 to an integer between 0 and 255.

    Args:
        x: A number that could be either float or integer.

    Returns:
        int: Converted integer value.
    """
    return int(round(x * 255)) if 0 <= x <= 1 and not float(x).is_integer() else int(x)


def hex_to_rgb(hex_color):
    """Convert a hex color string to RGB tuple.

    Args:
        hex_color (str): Hex color string (e.g., '#FF0000' or '#F00').

    Returns:
        tuple: RGB color tuple (r, g, b) with values between 0 and 255.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    r = int(h[0:2], 16)  # noqa: E702
    g = int(h[2:4], 16)  # noqa: E702
    b = int(h[4:6], 16)  # noqa: E702
    return r, g, b


def rgb_string_to_tuple(s):
    """Convert an RGB string to a tuple of integers.

    Args:
        s (str): RGB string (e.g., 'rgb(255,0,0)' or 'rgb(1.0,0,0)').

    Returns:
        tuple: RGB color tuple (r, g, b) with values between 0 and 255.
    """
    nums = list(map(float, re.findall(r"[\d.]+", s)))
    return tuple(_int_if_float(n) for n in nums[:3])


def to_rgba(color, alpha=0.3):
    """Convert various color formats to an RGBA string.

    Args:
        color: Color specification in one of these formats:
            - RGB tuple (0-255 or 0-1)
            - RGBA string
            - RGB string
            - Hex color string
        alpha (float, optional): Alpha value between 0 and 1. Defaults to 0.3.

    Returns:
        str: RGBA color string in format 'rgba(r,g,b,a)'.

    Raises:
        ValueError: If the color format is not supported.
    """
    if isinstance(color, (tuple, list)):
        r, g, b = color[:3]
        # handle 0..1 floats
        if max(r, g, b) <= 1:
            r, g, b = [_int_if_float(x) for x in (r, g, b)]
        return f"rgba({int(r)},{int(g)},{int(b)},{float(alpha)})"
    c = str(color).strip()
    if c.startswith("rgba"):
        parts = re.findall(r"[\d.]+", c)  #
        r, g, b, _ = parts[:4]
        return f"rgba({int(float(r))},{int(float(g))},{int(float(b))},{float(alpha)})"
    if c.startswith("rgb"):
        r, g, b = rgb_string_to_tuple(c)
        return f"rgba({r},{g},{b},{float(alpha)})"
    if c.startswith("#"):
        r, g, b = hex_to_rgb(c)
        return f"rgba({r},{g},{b},{float(alpha)})"
    raise ValueError(f"Unsupported color format: {color}")
