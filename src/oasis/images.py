from io import BytesIO
from pathlib import Path

from PIL import Image


def prepare_image(image_path, long_edge=448, already_prepared=False, policy="paper"):

    if type(long_edge) is not int or long_edge not in (448, 720):
        raise ValueError("long_edge must be 448 or 720")
    if policy not in ("paper", "archive-jpeg95"):
        raise ValueError("Image policy must be paper or archive-jpeg95")
    with Image.open(Path(image_path)) as source:
        image = source.convert("RGB")
    if already_prepared:
        return image
    ratio = long_edge / max(image.size)
    size = tuple(max(1, round(side * ratio)) for side in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    image.close()
    if policy == "paper":
        return resized
    with BytesIO() as buffer:
        resized.save(buffer, format="JPEG", quality=95, subsampling=0)
        resized.close()
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
