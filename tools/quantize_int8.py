"""Static INT8 (QDQ) quantization of a YOLO11 ONNX model. usage: quantize_int8.py in.onnx out.onnx size method imgdir  (method: minmax | entropy | percentile)"""
import sys, glob, re
import numpy as np
import onnx
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                       QuantType, quantize_static)

src, dst, size, method, imgdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
files = sorted(glob.glob(imgdir + "/*.jpg"))[:32]   # the last images are kept for evaluation

def load(path):
    im = Image.open(path).convert("RGB")
    s = min(size / im.width, size / im.height)
    nw, nh = round(im.width * s), round(im.height * s)
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None]

class Reader(CalibrationDataReader):
    def __init__(self, name):
        self.it = iter(files); self.name = name
    def get_next(self):
        f = next(self.it, None)
        return None if f is None else {self.name: load(f)}

import onnx.version_converter as vc
m = vc.convert_version(onnx.load(src), 17)
onnx.save(m, dst + ".opset17.onnx")
src = dst + ".opset17.onnx"
input_name = m.graph.input[0].name
# Keep the sensitive head layers in float: final cls/box convs and the DFL conv.
exclude = [n.name for n in m.graph.node if n.op_type == "Conv"
           and re.search(r"/model\.23/(cv[23]\.\d/cv[23]\.\d\.2/Conv|dfl/conv/Conv)$", n.name)]
print("excluding", len(exclude), "head convs; calibrating on", len(files), "images, method", method)

cm = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
      "percentile": CalibrationMethod.Percentile}[method]
extra = {"CalibPercentile": 99.99} if method == "percentile" else {}
quantize_static(src, dst, Reader(input_name), quant_format=QuantFormat.QDQ,
                op_types_to_quantize=["Conv"], per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                nodes_to_exclude=exclude, calibrate_method=cm, extra_options=extra)
print("wrote", dst)
