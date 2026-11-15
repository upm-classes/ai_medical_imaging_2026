"""Run this ONCE on your own computer (or in a Colab session) to build montgomery_256.npz (about 10 MB).
It downloads the Montgomery County chest X-ray set from the NLM server, downsamples to 256 x 256, merges the left and right lung masks
and stores a fixed stratified train/val/test split. Then upload montgomery_256.npz to your Google Drive.

READ FIRST: the dataset ReadMe asks users not to share the data outside their research group/organisation and to cite
Candemir et al. 2014 and Jaeger et al. 2014. Check the terms with NLM before you share this file (see the course notes).

Usage:
  python prepare_montgomery_subset.py                       # try to download from the NLM server
  python prepare_montgomery_subset.py NLM-MontgomeryCXRSet.zip   # or point it to a zip / folder you downloaded by hand
Needs: numpy, opencv-python(-headless), scikit-learn.
"""
import sys
from pathlib import Path
import numpy as np
import re, time, zipfile, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, cv2
from sklearn.model_selection import train_test_split

NLM_BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}   # the default Python user agent gets HTTP 403
def _open(url): return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60)

def _list_pngs(url):
    last = None
    for u in (url, url + "index.html"):
        try:
            html = _open(u).read().decode("utf8", "ignore")
            names = sorted(set(re.findall(r'href="([^"?#]+\.png)"', html)))
            if names: return [urllib.parse.unquote(n.split("/")[-1]) for n in names]
        except Exception as e: last = e
    raise RuntimeError(f"could not list {url}: {last!r}")

def download_montgomery(dest="data/MontgomerySet", workers=8, retries=3):
    """Download the 138 chest X-rays and the left/right lung masks from the NLM server (about 600 MB). Do NOT re-share the files."""
    names = _list_pngs(NLM_BASE + "CXR_png/")
    jobs = []
    for sub in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
        (Path(dest) / sub).mkdir(parents=True, exist_ok=True)
        jobs += [(NLM_BASE + sub + "/" + urllib.parse.quote(n), Path(dest) / sub / n) for n in names]
    def get(job):
        url, out = job
        if out.exists() and out.stat().st_size > 0: return
        for t in range(retries):
            try: out.write_bytes(_open(url).read()); return
            except Exception: time.sleep(2 * (t + 1))
        raise RuntimeError(f"failed to download {url}")
    with ThreadPoolExecutor(workers) as ex: list(ex.map(get, jobs))
    return Path(dest)

def find_root(path, workdir="data/_unzipped"):
    """Accept a folder or a .zip (e.g. NLM-MontgomeryCXRSet.zip downloaded by hand) and return the folder that contains CXR_png/."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z: z.extractall(workdir)
        path = workdir
    def n_png(d): return len([f for f in d.iterdir() if f.suffix.lower() == ".png"])
    cands = [p for p in path.rglob("*") if p.is_dir() and p.name.lower() == "cxr_png" and "__macosx" not in str(p).lower()]
    cands = [c for c in cands if n_png(c) > 0]
    if not cands:
        seen = sorted({str(p.relative_to(path)) for p in path.rglob("*") if p.is_dir()})[:15]
        raise FileNotFoundError(f"no folder named CXR_png with .png files under {path}. Folders found: {seen}")
    best = max(cands, key=n_png); print(f"using {best} ({n_png(best)} images)")
    return best.parent

def read_gray01(path):
    """Read a (possibly 16-bit) PNG and stretch the 0.5-99.5 percentile range to 0..1."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(path)
    if im.ndim == 3: im = im[..., 0]
    im = im.astype(np.float32); lo, hi = np.percentile(im, [0.5, 99.5])
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)

def build_arrays(root, size=256, seed=0):
    """Downsample images, MERGE the left and right lung masks, label TB (1) / normal (0) from the file name, make a fixed stratified split."""
    root = Path(root); files = sorted(f for f in (root / "CXR_png").iterdir() if f.suffix.lower() == ".png")
    if not files: raise FileNotFoundError(f"no PNG images in {root / 'CXR_png'}")
    imgs, masks, tb, ids = [], [], [], []
    for f in files:
        im = read_gray01(f)
        left = cv2.imread(str(root / "ManualMask/leftMask" / f.name), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(root / "ManualMask/rightMask" / f.name), cv2.IMREAD_UNCHANGED)
        merged = ((left > 0) | (right > 0)).astype(np.float32)                         # one mask: both lungs
        imgs.append(np.round(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8))
        masks.append((cv2.resize(merged, (size, size), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8))
        tb.append(int(f.stem.split("_")[-1])); ids.append(f.stem)
    n = len(files); tb = np.array(tb, np.uint8); idx = np.arange(n); k = int(round(0.15 * n))
    rest, test = train_test_split(idx, test_size=k, stratify=tb, random_state=seed)
    train, val = train_test_split(rest, test_size=k, stratify=tb[rest], random_state=seed)
    return dict(images=np.stack(imgs), masks=np.stack(masks), tb=tb, ids=np.array(ids), train_idx=np.sort(train), val_idx=np.sort(val), test_idx=np.sort(test))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    root = find_root(src) if src else download_montgomery("MontgomerySet")
    arrays = build_arrays(root, size=256)
    np.savez_compressed("montgomery_256.npz", **arrays)
    a = arrays
    print("saved montgomery_256.npz:", a["images"].shape, "images |", len(a["train_idx"]), "/", len(a["val_idx"]), "/", len(a["test_idx"]), "train/val/test | TB cases:", int(a["tb"].sum()))
    print("size on disk (MB):", round(Path("montgomery_256.npz").stat().st_size / 1e6, 1))
