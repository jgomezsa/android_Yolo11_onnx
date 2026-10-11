"""Compare fp32 vs quantized YOLO outputs on held-out images. usage: eval_quantized.py fp32.onnx int8.onnx size imgdir"""
import sys, glob, time
import numpy as np
import onnxruntime as ort
from PIL import Image

fp, q, size, imgdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
files = sorted(glob.glob(imgdir + "/*.jpg"))[32:]   # held out from calibration

def load(path):
    im = Image.open(path).convert("RGB")
    s = min(size / im.width, size / im.height)
    nw, nh = round(im.width * s), round(im.height * s)
    im = im.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (size, size), (114, 114, 114)); c.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return (np.asarray(c, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

def detect(sess, x, conf=0.25, iou=0.45):
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]   # [84, N]
    boxes, scores = out[:4].T, out[4:].T
    cls = scores.argmax(1); sc = scores.max(1); keep = sc > conf
    b, c, s = boxes[keep], cls[keep], sc[keep]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    order = np.argsort(-s); res = []
    for i in order:
        ok = True
        for j in res:
            if c[j] != c[i]: continue
            iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0]); ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            if iw > 0 and ih > 0:
                inter = iw * ih; ua = (xyxy[i, 2]-xyxy[i, 0])*(xyxy[i, 3]-xyxy[i, 1]) + (xyxy[j, 2]-xyxy[j, 0])*(xyxy[j, 3]-xyxy[j, 1]) - inter
                if inter / ua > iou: ok = False; break
        if ok: res.append(i)
    return [(int(c[i]), float(s[i]), xyxy[i]) for i in res]

def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih; return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

opts = ort.SessionOptions(); opts.log_severity_level = 3
sf = ort.InferenceSession(fp, opts, providers=["CPUExecutionProvider"])
sq = ort.InferenceSession(q, opts, providers=["CPUExecutionProvider"])
matched = tot = extra = 0; dscore = []
for f in files:
    x = load(f); a = detect(sf, x); b = detect(sq, x)
    used = set()
    for ca, sa, ba in a:
        tot += 1
        best = None
        for k, (cb, sb, bb) in enumerate(b):
            if k not in used and cb == ca and iou(ba, bb) > 0.5 and (best is None or sb > b[best][1]): best = k
        if best is not None: used.add(best); matched += 1; dscore.append(b[best][1] - sa)
    extra += len(b) - len(used)
print(f"{q.split('/')[-1]}: fp32 detections matched {matched}/{tot} ({100*matched/max(tot,1):.0f}%), "
      f"extra int8 detections {extra}, mean score delta {np.mean(dscore) if dscore else 0:+.3f}")
