import torch
import numpy as np
from disk_features.disk import DISK
import torch.nn.functional as F
from disk_features.disk.geom import distance_matrix

DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Image:
    def __init__(self, bitmap, fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    def to_image_coord(self, xys):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    def _interpolate(self, image, shape):
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    def _pad(self, image, shape):
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))

def init_model():
    state_dict = torch.load('./disk_features/pretrained/depth-save.pth', map_location=DEV)
    weights = state_dict['extractor']
    model = DISK(window=8, desc_dim=128)
    model.load_state_dict(weights)
    model = model.to(DEV)
    return model

model = init_model()

def extract_features(img):
    tensor = torch.from_numpy(img).to(torch.float32)
    bitmap = tensor.permute(2,0,1) / 255.
    image = Image(bitmap, 'image')
    image = image.resize_to((512, 1024))
    with torch.no_grad():
        batched_features = model.features(torch.stack([image.bitmap], 0).to(DEV, non_blocking=True), kind='nms', window_size=5, cutoff=0., n=None)
    features = batched_features.flat[0]
    features = features.to(DEV)
    kps_crop_space = features.kp.T
    kps_img_space, mask = image.to_image_coord(kps_crop_space)
    keypoints   = kps_img_space.detach().to('cpu').T[mask].numpy()
    descriptors = features.desc.detach().to('cpu')[mask].numpy()
    scores      = features.kp_logp.detach().to('cpu')[mask].numpy()
    order = np.argsort(scores)[::-1]
    keypoints   = keypoints[order]
    descriptors = descriptors[order]
    scores      = scores[order]
    return keypoints, descriptors

MAX_FULL_MATRIX = 10000**2

def _binary_to_index(binary_mask, ix2):
    return torch.stack([
        torch.nonzero(binary_mask, as_tuple=False)[:, 0],
        ix2
    ], dim=0)

def _ratio_one_way(dist_m, rt):
    val, ix = torch.topk(dist_m, k=2, dim=1, largest=False)
    ratio = val[:, 0] / val[:, 1]
    passed_test = ratio < rt
    ix2 = ix[passed_test, 0]

    return _binary_to_index(passed_test, ix2)

def _match_chunkwise(ds1, ds2, rt):
    chunk_size = MAX_FULL_MATRIX // ds1.shape[0]
    matches = []
    start = 0

    while start < ds2.shape[0]:
        ds2_chunk = ds2[start:start+chunk_size]
        dist_m = distance_matrix(ds1, ds2_chunk)
        one_way = _ratio_one_way(dist_m, rt)
        one_way[1] += start
        matches.append(one_way)
        start += chunk_size

    return torch.cat(matches, dim=1)

def _match(ds1, ds2, rt):
    size = ds1.shape[0] * ds2.shape[0]

    fwd = _match_chunkwise(ds1, ds2, rt)
    bck = _match_chunkwise(ds2, ds1, rt)
    bck = torch.flip(bck, (0, ))

    merged = torch.cat([fwd, bck], dim=1)
    unique, counts = torch.unique(merged, dim=1, return_counts=True)

    return unique[:, counts == 2]

def match(desc_1, desc_2, rt=1., u16=False):
    matched_pairs = _match(desc_1, desc_2, rt)
    matches = matched_pairs.cpu().numpy()

    if u16:
        matches = matches.astype(np.uint16)

    return matches

def match_features(cam1, cam2):
    kp0, des0 = cam1.getFeature()
    kp1, des1 = cam2.getFeature()
    des0 = torch.from_numpy(des0).to(torch.float32)
    des1 = torch.from_numpy(des1).to(torch.float32)
    matches = match(des0, des1, rt=1, u16=True)
    return kp0[matches[0]], kp1[matches[1]], matches[0], matches[1]

# def extract_features(imggray):
#     detect = cv2.SIFT_create()
#     descript = detect
#     kp = detect.detect(imggray, None)
#     kp,des = descript.compute(imggray, kp)
#     kp = np.float32([pt.pt for pt in kp])
#     return kp,des

# def match_feature(fea0, fea1):
#     kp0, des0 = fea0.getFeature()
#     kp1, des1 = fea1.getFeature()

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des0, des1, k=2)

#     good = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good.append(m)

#     index0 = np.int32([m.queryIdx for m in good])
#     index1 = np.int32([m.trainIdx for m in good])

#     return kp0[index0], kp1[index1], index0, index1