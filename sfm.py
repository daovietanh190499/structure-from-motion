import os
import cv2
import numpy as np
from tqdm import tqdm
import exifread
from disk_features.feature import extract_features, match_features
from scipy.optimize import least_squares

img_dir = '../dataset/gustav/'
images = sorted( filter( lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir) ) )
cameras = []
point_cloud = []
point_color = []

class Camera:
    def __init__(self, id, img, kp, desc, match2d3d):
        self.id = id
        self.img = img
        self.kp = kp
        self.desc = desc 
        self.match2d3d = match2d3d
        self.Rt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.reconstruct = False

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True
    
    def getRt(self):
        return self.Rt[:3,:3], self.Rt[:3, 3]

    def getRelativeRt(self, cam2):
        return cam2.Rt[:3,:3].T.dot(self.Rt[:3,:3]), cam2.Rt[:3, :3].T.dot(self.Rt[:3, 3] - cam2.Rt[:3, 3])
    
    def getP(self, K):
        return np.matmul(K, self.Rt)
    
    def getPos(self):
        pts = np.array([[0,0,0]]).T
        pts = self.Rt[:3,:3].T.dot(pts)- self.Rt[:3,3][:,np.newaxis]
        return pts[:,0]
    
    def getFeature(self):
        return (self.kp, self.desc)

def get_camera_intrinsic_params(images_dir):
    K = []
    h, w, c = cv2.imread(images_dir + os.listdir(images_dir)[1]).shape
    img = open(images_dir + os.listdir(images_dir)[1], 'rb')
    exif = exifread.process_file(img, details=False)
    exif = exif if 'EXIF FocalLengthIn35mmFilm' in exif else {'EXIF FocalLengthIn35mmFilm': exifread.classes.IfdTag(True, 'focal', list, [37.66], 1, 32)}
    image_width, image_height = (w, h) if w > h else (h, w)
    focal_length = (exif['EXIF FocalLengthIn35mmFilm'].values[0]/35)*image_width
    K.append([focal_length, 0, w/2])
    K.append([0, focal_length, h/2])
    K.append([0, 0, 1])
    return {'width': image_width, 'height': image_height}, np.array(K, dtype=float)

def triangulate(cam1, cam2, idx0, idx1, K):
    points_3d = cv2.triangulatePoints(cam1.getP(K), cam2.getP(K), cam1.kp[idx0].T, cam2.kp[idx1].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    point2d_ind = idx1[np.where(cam1.match2d3d[idx0] ==  -1)]
    for w, i in enumerate(idx0):
        if cam1.match2d3d[i] == -1:
            point_cloud.append(points_3d[w])
            point_color.append(cam1.img[int(cam1.kp[i][1]), int(cam1.kp[i][0]), :])
            cam1.match2d3d[i] = len(point_cloud) - 1
        cam2.match2d3d[idx1[w]] = cam1.match2d3d[i]
    point3d_ind = cam2.match2d3d[point2d_ind]
    x = np.hstack((cv2.Rodrigues(cam2.getRt()[0])[0].ravel(), cam2.getRt()[1].ravel(), np.array(point_cloud)[point3d_ind].ravel()))
    res = least_squares(calculate_reprojection_error, x, gtol=0.5, args=(K, cam2.kp[point2d_ind]))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape((len(point3d_ind), 3))
    for i, j in enumerate(point3d_ind): point_cloud[j] = point_3D[i]
    cam2.setRt(R, t.reshape((3,1)))

def to_ply(img_dir, point_cloud, colors, subfix = "_sparse.ply"):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    print(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix)
    if not os.path.exists(img_dir + '/Point_Cloud/'):
        os.makedirs(img_dir + '/Point_Cloud/')
    with open(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def calculate_reprojection_error(x, K, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    error = np.linalg.norm(point_2D - reprojected_point, axis=1)
    return error / len(reprojected_point)

exif, K = get_camera_intrinsic_params(img_dir)
# K = np.array([[718.8560/downscale, 0, 607.1928/downscale], [0, 718.8560/downscale, 185.2157/downscale], [0,0,1]])

j = 0
for i in tqdm(range(len(images))):
    if images[i].split('.')[-1] in ['JPG', 'jpg', 'PNG', 'png', 'RAW', 'raw']:
        img = cv2.imread(img_dir + images[i])
        if img.shape[1] != exif['width'] or img.shape[0] != exif['height']:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        kp, des = extract_features(img)
        cameras.append(Camera(images[i], img.copy(), kp, des, np.ones((len(kp),), dtype='int32')*-1))
        if j > 0:
            pts0_, pts1_, idx0, idx1 = match_features(cameras[j-1], cameras[j])
            E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
            idx0, idx1 = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1]
            _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)
            if j != 1:
                match = np.int32(np.where(cameras[j-1].match2d3d[idx0] != -1)[0])
                if len(match) < 8: continue
                ret, rvecs, t, inliers = cv2.solvePnPRansac(np.float32(point_cloud)[cameras[j-1].match2d3d[idx0[match]]], cameras[j].kp[idx1[match]], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
                R, _ = cv2.Rodrigues(rvecs)
            cameras[j].setRt(R, t)
            triangulate(cameras[j-1], cameras[j], idx0, idx1, K)
        j += 1

to_ply(img_dir, np.array(point_cloud), np.array(point_color))
to_ply(img_dir, np.array([cam.getPos() for cam in cameras]), np.ones_like(np.array([cam.getPos() for cam in cameras]))*255, '_campos.ply')