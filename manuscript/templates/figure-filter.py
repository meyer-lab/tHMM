#! /usr/bin/env python3

from mimetypes import guess_type
from os.path import splitext
from pandocfilters import toJSONFilter, Image

fmt_to_option = {
    "latex": ("--export-pdf", "pdf"),
    "docx": ("--export-eps", "eps")
}


def svg_to_any(key, value, fmt, meta):
    if key == 'Image':
        attrs, alt, [src, title] = value
        mimet, _ = guess_type(src)
        option = fmt_to_option.get(fmt)
        if mimet == 'image/svg+xml' and option:
            base_name, _ = splitext(src)
            eps_name = base_name + "." + option[1]
            return Image(attrs, alt, [eps_name, title])


if __name__ == "__main__":
    toJSONFilter(svg_to_any)
