from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
import math

import cv2
import numpy as np

from .config import UserError, VideoSettings
from .records import Candidate, Evidence, Prepared, Track, TrackWindow, View
from .storage import Job, write_atomic
from .video import frames_for, sample_indices, scan
from .weights import ensure_weights


WINDOW_MS = 10_000
FACE_LINK = 0.47
FACE_MARGIN = 0.055


def resize_cap(image, pixels):
    height, width = image.shape[:2]
    scale = min(1.0, pixels / max(height, width))
    if scale == 1:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def encode(image) -> bytes:
    success, encoded = cv2.imencode(".jpg", resize_cap(image, 1600), [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not success:
        raise UserError("A selected frame could not be encoded as a JPEG.")
    return encoded.tobytes()


def image_quality(image) -> float:
    gray = cv2.cvtColor(resize_cap(image, 400), cv2.COLOR_BGR2GRAY)
    focus = min(1.0, math.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 6)
    usable_exposure = np.mean((gray > 24) & (gray < 246))
    pixels = min(1.0, min(image.shape[:2]) / 320)
    return float(0.50 * focus + 0.25 * usable_exposure + 0.25 * pixels)


def signature(image) -> tuple[list[float], list[float]]:
    height, width = image.shape[:2]
    fabric = image[height * 3 // 10:max(height * 8 // 10, 1), width // 4:max(width * 3 // 4, 1)]
    hsv = cv2.cvtColor(resize_cap(fabric, 160), cv2.COLOR_BGR2HSV)
    color = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    color /= max(1, color.sum())
    thumb = cv2.cvtColor(cv2.resize(fabric, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    texture = thumb.astype(np.float32).ravel() / 255
    return color.round(5).tolist(), texture.round(5).tolist()


def similarity(a: View, b: View) -> float:
    colors = sum(min(i, j) for i, j in zip(a.color_signature, b.color_signature))
    brightness = 1 - sum(abs(i - j) for i, j in zip(a.texture_signature, b.texture_signature)) / 64
    return float(0.65 * colors + 0.35 * brightness)


def choose_views(views: list[View], limit: int) -> list[View]:
    best = max(views, key=lambda view: (view.person.score, -view.person.frame_index))
    selected = [best]
    if limit >= 2 and len(views) > 1:
        wide = max(views, key=lambda view: (view.body_ratio, view.person.score))
        if wide != best:
            selected.append(wide)
    while len(selected) < min(limit, len(views)):
        spread = max(v.person.timestamp_ms for v in views) - min(v.person.timestamp_ms for v in views)

        def priority(view):
            distance = min(abs(p.person.timestamp_ms - view.person.timestamp_ms) for p in selected)
            novelty = 1 - max(similarity(p, view) for p in selected)
            return 0.55 * view.person.score + 0.30 * distance / max(spread, 1) + 0.15 * novelty

        selected.append(max((view for view in views if view not in selected), key=priority))
    return sorted(selected, key=lambda view: view.person.timestamp_ms)


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


@dataclass
class Face:
    vector: np.ndarray
    portrait: bytes
    quality: float
    time_ms: int


@dataclass
class Sample:
    view: View
    photo: bytes


@dataclass
class TrackBuilder:
    id: str
    shot_id: str
    start_ms: int
    end_ms: int
    visible_ms: int = 0
    area_sum: float = 0
    count: int = 0
    faces: list[Face] = field(default_factory=list)
    window_index: int = -1
    samples: dict[tuple[int, str], Sample] = field(default_factory=dict)
    windows: list[TrackWindow] = field(default_factory=list)

    def flush(self, job):
        unique = {sample.view.person.id: sample for sample in self.samples.values()}
        if not unique:
            return
        selected = choose_views([sample.view for sample in unique.values()], 6)
        for view in selected:
            write_atomic(job.path / view.person.path, unique[view.person.id].photo)
        self.windows.append(TrackWindow(index=self.window_index, views=selected))
        self.samples.clear()

    def add(self, job, crop, index, ms, shot_start, area, visible_ms, face):
        self.end_ms = ms
        self.visible_ms += visible_ms
        self.area_sum += area
        self.count += 1
        if face:
            near = next((previous for previous in self.faces if previous.time_ms // 650 == ms // 650), None)
            if near is None or face.quality > near.quality:
                if near is not None:
                    self.faces = [previous for previous in self.faces if previous is not near]
                self.faces.append(face)
                self.faces.sort(key=lambda item: item.quality, reverse=True)
                self.faces = self.faces[:10]

        window_index, offset = divmod(ms - shot_start, WINDOW_MS)
        if window_index != self.window_index:
            self.flush(job)
            self.window_index = window_index

        quality = image_quality(crop)
        # Taller views are useful for trousers and hemlines, even if slightly less sharp.
        ratio = min(3.0, crop.shape[0] / crop.shape[1])
        slot = offset // 500
        choices = {
            (slot, "clear"): lambda v: v.person.score,
            (slot, "wide"): lambda v: v.body_ratio * 0.27 + v.person.score,
        }
        color, texture = signature(crop)
        evidence_id = f"{self.id}-f{index:07}"
        evidence = Evidence(
            id=evidence_id, shot_id=self.shot_id, frame_index=index, timestamp_ms=ms,
            kind="person", path=f"evidence/{evidence_id}.jpg", score=round(quality, 4),
        )
        view = View(
            person=evidence, color_signature=color, texture_signature=texture,
            body_ratio=ratio, clear_upper=face is not None,
        )
        wins = [
            key for key, rank in choices.items()
            if key not in self.samples or rank(view) > rank(self.samples[key].view)
        ]
        if wins:
            sample = Sample(view, encode(crop))
            for key in wins:
                self.samples[key] = sample

    def finish(self, job):
        self.flush(job)
        flags = []
        vector = None
        if self.faces:
            vectors = np.stack([face.vector for face in self.faces])
            similarities = vectors @ vectors.T
            if len(vectors) >= 2 and np.min(similarities) < 0.18:
                flags.append("Face matches disagree in this track. Check the views before including it.")
            else:
                vector = unit(np.average(vectors, axis=0, weights=[face.quality for face in self.faces]))
        else:
            flags.append("No clear face in this shot. Assign the track manually if it belongs to a lead.")
        track = Track(
            id=self.id, shot_id=self.shot_id, start_ms=self.start_ms, end_ms=self.end_ms,
            visible_ms=self.visible_ms, prominence=self.area_sum / max(self.count, 1),
            windows=self.windows, flags=flags,
        )
        return track, vector, self.faces[:2]


def face_assignments(image, boxes, detector, recognizer, ms):
    if not len(boxes):
        return {}
    reduced = resize_cap(image, 1280)
    height, width = reduced.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(reduced)
    if detections is None:
        return {}
    detections[:, :14] *= np.tile([image.shape[1] / width, image.shape[0] / height], 7)
    proposals = defaultdict(list)
    for row in detections:
        x, y, w, h = row[:4]
        if min(w, h) < 27:
            continue
        cx, cy = x + w / 2, y + h / 2
        scores = []
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.55 * (y2 - y1):
                nx = (cx - x1) / (x2 - x1)
                ny = (cy - y1) / (y2 - y1)
                scores.append((abs(nx - 0.5) + 1.3 * abs(ny - 0.15), index))
        scores.sort()
        if not scores or (len(scores) > 1 and scores[1][0] - scores[0][0] < 0.09):
            continue
        proposals[scores[0][1]].append(row)
    faces = {}
    for index, rows in proposals.items():
        if len(rows) != 1:
            continue
        row = rows[0]
        cropped = recognizer.alignCrop(image, row)
        quality = 0.68 * image_quality(cropped) + 0.32 * min(1.0, row[2] / 120)
        if quality >= 0.37:
            vector = unit(recognizer.feature(cropped).flatten())
            faces[index] = Face(vector, encode(cropped), float(quality), ms)
    return faces


def person_boxes(detections, image):
    detections = detections[detections.class_id == 1]
    boxes = detections.xyxy
    height, width = image.shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return detections[(boxes[:, 2] - boxes[:, 0] >= 22) & (boxes[:, 3] - boxes[:, 1] >= 30)].with_nms(0.65)


@dataclass
class Cluster:
    members: list[tuple[Track, np.ndarray, list[Face]]]
    centroid: np.ndarray


def compatible(track, cluster):
    return all(
        other.shot_id != track.shot_id or min(other.end_ms, track.end_ms) - max(other.start_ms, track.start_ms) < 250
        for other, _, _ in cluster.members
    )


def group_cast(job, found):
    clusters: list[Cluster] = []
    ranked = sorted(
        (entry for entry in found if entry[1] is not None),
        key=lambda item: (-item[2][0].quality, -item[0].visible_ms, item[0].id),
    )
    for track, vector, faces in ranked:
        comparisons = sorted(
            ((float(vector @ cluster.centroid), i) for i, cluster in enumerate(clusters)),
            reverse=True,
        )
        candidates = [(score, i) for score, i in comparisons if score >= FACE_LINK and compatible(track, clusters[i])]
        if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < FACE_MARGIN):
            clusters.append(Cluster([(track, vector, faces)], vector))
            continue
        cluster = clusters[candidates[0][1]]
        cluster.members.append((track, vector, faces))
        cluster.centroid = unit(np.mean([item[1] for item in cluster.members], axis=0))

    def cluster_rank(cluster):
        tracks = [item[0] for item in cluster.members]
        duration = sum(track.visible_ms for track in tracks)
        shots = len({track.shot_id for track in tracks})
        return duration / 1000 + 0.35 * shots + max(track.prominence for track in tracks)

    output = []
    for index, cluster in enumerate(sorted(clusters, key=cluster_rank, reverse=True), 1):
        candidate_id = f"cast-{index:03}"
        portraits = []
        options = sorted(
            (face for _, _, faces in cluster.members for face in faces),
            key=lambda face: (-face.quality, face.time_ms),
        )
        used = set()
        for face in options:
            if face.time_ms // 1000 in used:
                continue
            used.add(face.time_ms // 1000)
            relative = f"cast/{candidate_id}-{len(portraits) + 1}.jpg"
            write_atomic(job.path / relative, face.portrait)
            portraits.append(relative)
            if len(portraits) == 3:
                break
        output.append(Candidate(
            id=candidate_id, track_ids=sorted(track.id for track, _, _ in cluster.members),
            shots=len({track.shot_id for track, _, _ in cluster.members}),
            visible_ms=sum(track.visible_ms for track, _, _ in cluster.members), portraits=portraits,
        ))
    return output


def make_detector(device_name, path, progress):
    import torch
    from rfdetr import RFDETRSmall

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    progress(f"Loading person detection · {device_name.upper()}", 0.18)
    if device_name != "mps":
        return RFDETRSmall(pretrain_weights=str(path), device=device_name)
    # Run one image through the Apple backend before starting a long job.
    try:
        model = RFDETRSmall(pretrain_weights=str(path), device=device_name)
        model.predict(np.zeros((256, 256, 3), dtype=np.uint8), threshold=0.25)
    except (NotImplementedError, RuntimeError) as exc:
        progress(f"Apple GPU unavailable for this model. Using CPU ({type(exc).__name__}).", 0.18)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        return RFDETRSmall(pretrain_weights=str(path), device="cpu")
    return model


def prepare(job: Job, settings: VideoSettings, progress) -> Prepared:
    from trackers import ByteTrackTracker

    with job.lock():
        existing = job.read("prepared", Prepared)
        if existing is not None:
            progress("Video already scanned", 1.0)
            return existing
        paths = ensure_weights(progress)
        timeline = scan(job.source, lambda message, fraction: progress(message, 0.18 * fraction))
        model = make_detector(settings.device, paths["person"], progress)
        faces = cv2.FaceDetectorYN.create(str(paths["face"]), "", (320, 320), 0.85, 0.3, 500)
        recognition = cv2.FaceRecognizerSF.create(str(paths["recognition"]), "")
        samples = {
            index: shot for shot in timeline.shots
            for index in sample_indices(shot, timeline.times_ms, settings.sample_fps)
        }
        found = []
        decoded = 0
        for shot, frames in groupby(frames_for(job.source, set(samples)), key=lambda frame: samples[frame[0]]):
            tracker = ByteTrackTracker(
                frame_rate=30.0, lost_track_buffer=12, track_activation_threshold=0.40,
                high_conf_det_threshold=0.40, minimum_consecutive_frames=1,
            )
            builders = {}
            previous_ms = timeline.times_ms[shot.start]
            for index, image in frames:
                ms = timeline.times_ms[index]
                next_ms = timeline.times_ms[index + 1] if index + 1 < len(timeline.times_ms) else timeline.duration_ms
                visible_ms = max(1, min(250, max(ms - previous_ms, next_ms - ms)))
                previous_ms = ms
                detected = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=0.15)
                tracked = tracker.update(person_boxes(detected, image), timestamp=ms / 1000)
                if len(tracked):
                    # ByteTrack uses -1 for unmatched detections, which may be different people each frame.
                    tracked = tracked[tracked.tracker_id >= 0]
                boxes = tracked.xyxy.round().astype(int)
                matched_faces = face_assignments(image, boxes, faces, recognition, ms)
                height, width = image.shape[:2]
                for position, (box, tracker_id) in enumerate(zip(boxes, tracked.tracker_id if len(tracked) else [])):
                    x1, y1, x2, y2 = box
                    area = float((x2 - x1) * (y2 - y1) / (width * height))
                    if tracker_id not in builders:
                        builders[tracker_id] = TrackBuilder(f"{shot.id}-p{tracker_id:03}", shot.id, ms, ms)
                    builders[tracker_id].add(
                        job, image[y1:y2, x1:x2], index, ms, timeline.times_ms[shot.start],
                        area, visible_ms, matched_faces.get(position),
                    )
                decoded += 1
                if decoded % 12 == 0:
                    progress(f"Finding people · {decoded:,} / {len(samples):,} frames", 0.20 + 0.78 * decoded / len(samples))
            found.extend(builder.finish(job) for builder in builders.values())

        if not found:
            raise UserError("No people could be tracked. Try a clip with visible people or set video.sample_fps = 0 for a short clip.")
        prepared = Prepared(
            duration_ms=timeline.duration_ms, shot_count=len(timeline.shots), sampled_frames=decoded,
            tracks=[result[0] for result in found],
            candidates=group_cast(job, found),
        )
        job.save("prepared", prepared)
        progress("Video scanned. Confirm the primary cast.", 1.0)
        return prepared
