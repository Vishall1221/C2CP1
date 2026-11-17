# Model weights

Git LFS stores these three unmodified checkpoints. Run `git lfs pull`
from the repository to fetch the binaries. C2C verifies each file against
the SHA-256 checksum in [c2c/weights.py](../c2c/weights.py) before using it.

| File | Origin | License |
| --- | --- | --- |
| `rf-detr-small.pth` | [Roboflow RF-DETR Small checkpoint](https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth) | [Apache 2.0](licenses/RF-DETR-Apache-2.0.txt) |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet](https://huggingface.co/opencv/face_detection_yunet) | [MIT, Shiqi Yu](licenses/YuNet-MIT.txt) |
| `face_recognition_sface_2021dec.onnx` | [OpenCV Zoo SFace](https://huggingface.co/opencv/face_recognition_sface) | [Apache 2.0](licenses/SFace-Apache-2.0.txt) |

Roboflow publishes the RF-DETR Small weights under Apache 2.0 in its
[pretrained model documentation](https://rfdetr.roboflow.com/latest/learn/pretrained/).
SFace was contributed by Yaoyao Zhong and converted to ONNX by Chengrui Wang,
as credited in the [OpenCV Zoo SFace README](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md).
