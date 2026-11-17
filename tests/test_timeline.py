from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from c2c.video import frames_for, sample_indices, scan


class TimelineTest(unittest.TestCase):
    def test_variable_rate_timestamps_come_from_mp4(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "variable.mp4"
            with av.open(str(source), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width = 160
                stream.height = 96
                stream.pix_fmt = "yuv420p"
                stream.codec_context.time_base = Fraction(1, 1000)
                for tick in (1000, 1045, 1151, 1426, 1620, 1658, 1722, 1861):
                    image = np.full((96, 160, 3), 125, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    frame.pts = tick
                    frame.time_base = Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            result = scan(source, lambda *_: None)
            self.assertEqual(len(result.shots), 1)
            self.assertEqual(result.times_ms, [0, 45, 151, 426, 620, 658, 722, 861])
            indices = sample_indices(result.shots[0], result.times_ms, 6)
            self.assertEqual([result.times_ms[i] for i in indices], [0, 426, 620, 722, 861])
            self.assertEqual([i for i, _ in frames_for(source, set(indices))], indices)
