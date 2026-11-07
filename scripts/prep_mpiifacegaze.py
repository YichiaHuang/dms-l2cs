"""MPIIFaceGaze -> GazeHub "FaceBased/MPIIFaceGaze" layout, as consumed by L2CS-Net's Mpiigaze dataset.

Port of GazeHub's data_processing_mpii.py + data_processing_core.py (phi-ai.buaa.edu.cn/Gazehub).
Differences from the GazeHub original:
  * which-eye flag is taken from column 28 of pXX.txt (MPIIFaceGaze ships it), so the separate
    MPIIGaze "sample list for eye image" download is not needed;
  * no debug print in HeadTo2d; persons are processed in parallel.
Output: <out>/Image/pXX/{face,left,right}/N.jpg and <out>/Label/pXX.label with header
  Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin
2DGaze = (yaw, pitch) in radians -- L2CS reads [0] as "pitch" and [1] as "yaw" (names swapped in that repo).

  python prep_mpiifacegaze.py --raw ~/datasets/raw/MPIIFaceGaze/MPIIFaceGaze --out ~/datasets/MPIIFaceGaze
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import scipy.io as sio

NEW_FOCAL, NEW_DIST, IMSIZE = 960, 600, (224, 224)


class Norm:
    def __init__(self, center, target, headrotvec, cam):
        self.center = np.asarray(center, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.cam = np.asarray(cam, dtype=np.float64)
        self.curdist = np.linalg.norm(self.center)
        self.headrot = cv2.Rodrigues(np.asarray(headrotvec, dtype=np.float64))[0]
        self.S = np.diag([1.0, 1.0, NEW_DIST / self.curdist])
        z = self.center / self.curdist
        y = np.cross(z, self.headrot[:, 0]); y /= np.linalg.norm(y)
        x = np.cross(y, z); x /= np.linalg.norm(x)
        self.R = np.array([x, y, z])
        self.C = np.array([[NEW_FOCAL, 0, IMSIZE[0] / 2], [0, NEW_FOCAL, IMSIZE[1] / 2], [0, 0, 1]], dtype=np.float64)
        self.M = self.S @ self.R
        self.W = self.C @ self.M @ np.linalg.inv(self.cam)
        self.gaze = self.target - self.center

    def image(self, im):
        self.im = cv2.warpPerspective(im, self.W, IMSIZE)
        return self.im

    def new_pos(self, p):
        r = np.append(np.asarray(p, dtype=np.float64), 1.0) @ self.W.T
        return np.array([r[0] / r[2], r[1] / r[2]])

    def crop_eye(self, lc, rc):
        x, y = list(zip(lc, rc))
        cx, cy = np.mean(x), np.mean(y)
        w = abs(x[0] - x[1]) * 1.5
        h = 36 * (w / 60)
        x1 = [max(cx - w / 2, 0), max(cy - h / 2, 0)]
        x2 = [min(x1[0] + w, IMSIZE[0]), min(x1[1] + h, IMSIZE[1])]
        im = self.im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        if im.size == 0:
            im = np.zeros((36, 60, 3), np.uint8)
        return cv2.resize(im, (60, 36))

    def gaze_vec(self, scale=True):
        g = (self.M if scale else self.R) @ self.gaze
        return g / np.linalg.norm(g)

    def head_vec(self):
        return cv2.Rodrigues(self.M @ self.headrot)[0].T[0]

    def coord(self, c):
        return (self.M @ np.reshape(c, (3, 1))).flatten()

    def params(self):
        return cv2.Rodrigues(self.R)[0].flatten(), np.diagonal(self.S)


def gaze_to_2d(g):
    return np.array([np.arctan2(-g[0], -g[2]), np.arcsin(-g[1])])


def head_to_2d(h):
    v = cv2.Rodrigues(h)[0][:, 2]
    return np.array([np.arctan2(v[0], v[2]), np.arcsin(v[1])])


def gaze_flip(g):
    return np.array([-g[0], g[1], g[2]])


def head_flip(h):  # GazeHub's HeadFlip returns the *unflipped* matrix (bug); kept identical for parity.
    return cv2.Rodrigues(cv2.Rodrigues(h)[0])[0].T[0]


def equalize(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(im)


def fmt(a):
    return ",".join(str(float(v)) for v in np.asarray(a).flatten())


def process_person(args):
    raw, out, person = args
    im_root = os.path.join(raw, person)
    cam = sio.loadmat(os.path.join(im_root, "Calibration", "Camera.mat"))["cameraMatrix"]
    with open(os.path.join(im_root, f"{person}.txt")) as f:
        lines = [l.strip().split(" ") for l in f if l.strip()]
    for sub in ("face", "left", "right"):
        os.makedirs(os.path.join(out, "Image", person, sub), exist_ok=True)
    os.makedirs(os.path.join(out, "Label"), exist_ok=True)
    n_ok = 0
    with open(os.path.join(out, "Label", f"{person}.label"), "w") as lab:
        lab.write("Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin\n")
        for count, parts in enumerate(lines):
            im_info, which_eye = parts[0], parts[-1]
            a = np.array(parts[1:-1], dtype=np.float32)
            assert a.shape[0] == 26 and which_eye in ("left", "right"), (person, parts[:2], a.shape, which_eye)
            im = cv2.imread(os.path.join(im_root, im_info))
            if im is None:
                print(f"[{person}] missing image {im_info}", file=sys.stderr)
                continue
            llc, lrc, rlc, rrc = a[2:4], a[4:6], a[6:8], a[8:10]
            norm = Norm(center=a[20:23], target=a[23:26], headrotvec=a[14:17], cam=cam)
            im_face = norm.image(im)
            im_left = equalize(norm.crop_eye(norm.new_pos(llc), norm.new_pos(lrc)))
            im_right = equalize(norm.crop_eye(norm.new_pos(rlc), norm.new_pos(rrc)))
            gaze = norm.gaze_vec(scale=True)
            head = norm.head_vec()
            origin = norm.coord(a[20:23])
            rvec, svec = norm.params()
            if which_eye == "right":
                im_face = cv2.flip(im_face, 1)
                im_left, im_right = cv2.flip(im_right, 1), cv2.flip(im_left, 1)
                gaze = gaze_flip(gaze)
                head = head_flip(head)
                origin[0] = -origin[0]
            g2d, h2d = gaze_to_2d(gaze), head_to_2d(head)
            k = str(count + 1)
            cv2.imwrite(os.path.join(out, "Image", person, "face", k + ".jpg"), im_face)
            cv2.imwrite(os.path.join(out, "Image", person, "left", k + ".jpg"), im_left)
            cv2.imwrite(os.path.join(out, "Image", person, "right", k + ".jpg"), im_right)
            lab.write(" ".join([f"{person}/face/{k}.jpg", f"{person}/left/{k}.jpg", f"{person}/right/{k}.jpg",
                                im_info, which_eye, fmt(gaze), fmt(head), fmt(g2d), fmt(h2d),
                                fmt(rvec), fmt(svec), fmt(origin)]) + "\n")
            n_ok += 1
    return person, len(lines), n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing p00..p14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    persons = sorted(d for d in os.listdir(args.raw) if d.startswith("p") and os.path.isdir(os.path.join(args.raw, d)))
    print("persons:", persons)
    with Pool(args.workers) as pool:
        for person, n, ok in pool.imap_unordered(process_person, [(args.raw, args.out, p) for p in persons]):
            print(f"{person}: {ok}/{n} images", flush=True)


if __name__ == "__main__":
    main()
