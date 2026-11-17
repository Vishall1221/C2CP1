from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
import ssl
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import Request, urlopen
import os

from filelock import FileLock
import truststore

from .config import ROOT, UserError


MODELS = {
    "person": (
        "rf-detr-small.pth",
        "https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "sha256", "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33",
    ),
    "face": (
        "face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "sha256", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "recognition": (
        "face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "sha256", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def matches(path: Path, algorithm: str, expected: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return file_digest(handle, algorithm).hexdigest() == expected


def ensure_weights(progress) -> dict[str, Path]:
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    found = {}
    with FileLock(directory / ".weights.lock", timeout=300):
        for key, (name, url, algorithm, checksum) in MODELS.items():
            path = directory / name
            if not matches(path, algorithm, checksum):
                progress(f"Downloading {name} · first run only", 0.0)
                download(path, url, algorithm, checksum)
            found[key] = path
    return found


def download(path: Path, url: str, algorithm: str, checksum: str) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            request = Request(url, headers={"User-Agent": "C2C-Phase-1/0.1"})
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            with urlopen(request, timeout=90, context=context) as response:
                copyfileobj(response, temporary)
            temporary.close()
            if not matches(temp_path, algorithm, checksum):
                raise UserError(f"Checksum mismatch for {path.name}. Retry the model download.")
            os.replace(temp_path, path)
        except OSError as exc:
            cause = exc.reason if isinstance(exc, URLError) else exc
            if isinstance(cause, ssl.SSLCertVerificationError):
                raise UserError(
                    f"Certificate verification failed while downloading {path.name}: {cause}. "
                    "On a managed network, check with IT that the network's certificate is trusted "
                    "by your operating system, or retry on another trusted network."
                ) from exc
            raise UserError(f"Download failed for {path.name}: {exc}. Check the network and retry.") from exc
        finally:
            temporary.close()
            temp_path.unlink(missing_ok=True)
