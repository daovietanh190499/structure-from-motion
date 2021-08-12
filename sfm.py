import os
import cv2
import re
import numpy as np
from tqdm import tqdm
from tinydb import TinyDB, Query
from PIL import Image
from PIL.ExifTags import TAGS

class Point3d:
    def __init__(self, x, y, z, color=np.array([0,0,0])):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.triangulate = False
    
    def fromNumpy(self, pt):
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
        self.triangulate = True
    
    def numpy(self):
        return np.array([self.x, self.y, self.z])

class Camera:
    def __init__(self, id, img, kp, desc, match2d3d):
        self.id = id
        self.kp = kp
        self.desc = desc 
        self.match2d3d = match2d3d
        self.Rt = None
        self.reconstrucable = False
        self.reconstruct = False
        self.color = []
        for pt in kp:
            a,b = pt.pt
            self.color.append(img[int(b),int(a),:])

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True
    
    def getP(self, K):
        return np.matmul(K, self.Rt)
    
    def getPos(self):
        pts = np.array([[0,0,0]]).T
        pts = self.Rt[:3,:3].T.dot(pts)- self.Rt[:3,3][:,np.newaxis]
        return pts[:,0]
    
    def getFeature(self):
        return (self.kp, self.desc)
    
    def get3dIdx(self):
        return self.match2d3d[self.match2d3d != -1]

    def countMatch(self, other):
        return np.sum((np.in1d(self.match2d3d, other.match2d3d))*(self.match2d3d != -1))

    def getMatch(self, other):
        idx0 = np.where((np.in1d(self.match2d3d, other.match2d3d))*(self.match2d3d != -1))[0]
        idx1 = np.where((np.in1d(other.match2d3d, self.match2d3d))*(other.match2d3d != -1))[0]
        pts0 = np.array([self.kp[i].pt for i in idx0])
        pts1 = np.array([other.kp[i].pt for i in idx1])
        pts3d0 = self.match2d3d[idx0]
        pts3d1 = other.match2d3d[idx1]
        pts3d0_idx = np.argsort(pts3d0)
        pts3d1_idx = np.argsort(pts3d1)
        return pts0[pts3d0_idx], pts1[pts3d1_idx], pts3d0[pts3d0_idx]

def get_camera_intrinsic_params(images_dir):
    K = []
    available_path = []
    for path in os.listdir(images_dir):
        img = Image.open(images_dir + path)
        exif = {}
        try:
            exifdata = img.getexif()
        except:
            continue
        # print(exifdata)
        if exifdata and exifdata != '':
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                # decode bytes 
                if tag == 'FocalLength' or tag == 'Make' or tag == 'Model' or tag == 'ExifImageWidth' or tag == 'ExifImageHeight':
                    print(f"{tag:25}: {data}")
                    exif[tag] = data
            break
        else:
            continue
    
    if 'ExifImageWidth' not in exif.keys():
        exif['ExifImageWidth'] = img.size[0]
        exif['ExifImageHeight'] = img.size[1]
    if 'FocalLength' not in exif.keys():
        # H, inliers = cv2.findHomography(x1, x2, cv2.RANSAC, threshold)
        exif['FocalLength'] = 0.85
        sensor_width = 1
    else:
        if isinstance(exif['FocalLength'], tuple):
            exif['FocalLength'] = exif['FocalLength'][0]/exif['FocalLength'][1]
        exif['Model'] = exif['Model'].replace(exif['Make'].split(' ')[0], ' ')
        Camera = Query()
        db = TinyDB('cameras.json')
        res = db.search(Camera.make.search(exif['Make'].split(' ')[0], flags=re.IGNORECASE) & 
                        Camera.model.search(exif['Model'].strip(), flags=re.IGNORECASE))
        print(res[0])
        sensor_width = res[0]['sensor_width']
    
    for path in os.listdir(images_dir):
        img = Image.open(images_dir + path)
        if img.size[1] == exif['ExifImageHeight'] and img.size[0] == exif['ExifImageWidth']:
            available_path.append(path)
    
    focal_length = (exif['FocalLength']/sensor_width)*exif['ExifImageWidth']
    K.append([focal_length, 0, exif['ExifImageWidth']/2])
    K.append([0, focal_length, exif['ExifImageHeight']/2])
    K.append([0, 0, 1])
    K = np.array(K, dtype=float)

    return available_path, K

def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img

def extract_features(imggray, convert_gray):
    if convert_gray:
        imggray = cv2.cvtColor(imggray, cv2.COLOR_BGR2GRAY)
    detect = cv2.SIFT_create()
    kp = detect.detect(imggray, None)
    kp,des = detect.compute(imggray, kp)
    return kp,des

def match_feature(fea0, fea1):
    kp0, des0 = fea0.getFeature()
    kp1, des1 = fea1.getFeature()

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    index0 = np.int32([m.queryIdx for m in good])
    index1 = np.int32([m.trainIdx for m in good])

    return pts0, pts1, index0, index1

def triangulate(P1, P2, pts0, pts1):
    points_3d = cv2.triangulatePoints(P1, P2, pts0.T, pts1.T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    return points_3d

def to_ply(path, img_dir, point_cloud, colors, densify, campos):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    #print( verts.shape)
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
    print(path + '/Point_Cloud/' + img_dir.split('/')[-2] + '_*.ply')
    if not densify and not campos:
        with open(path + '/Point_Cloud/' + img_dir.split('/')[-2] + '_sparse.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    elif not campos:
        with open(path + '/Point_Cloud/' + img_dir.split('/')[-2] + '_dense.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open(path + '/Point_Cloud/' + img_dir.split('/')[-2] + '_campos.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

path = os.getcwd()
img_dir = path + '/data/vnu1/'
downscale = 1
convert_gray = False

available_path, K = get_camera_intrinsic_params(img_dir)
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# available_path = ['000.jpg', '001.jpg', '060.jpg', '090.jpg', '120.jpg', '150.jpg', '151.jpg', '152.jpg', '153.jpg', '154.jpg', '155.jpg', '156.jpg', '157.jpg', '158.jpg']

cameras = []
for i in tqdm(range(len(available_path))):
    img = Image.open(img_dir + available_path[i])
    img = img_downscale(np.array(img)[:,:,::-1], downscale)
    kp, des = extract_features(img, convert_gray)
    cameras.append(Camera(available_path[i], img, kp, des, np.ones((len(kp),), dtype='int32')*-1))

point_cloud = []
old_index1 = np.array([])
i = 0
j = 1
with tqdm(total=len(cameras)-1) as pbar:
    while i != len(cameras) - 1 and i + j < len(cameras):
        pts0, pts1, index0, index1 = match_feature(cameras[i], cameras[i + j])
        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.99)
        if not isinstance(mask, np.ndarray):
            break
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
        index0 = index0[mask.ravel() == 1]
        index1 = index1[mask.ravel() == 1]
        match2d3d = np.where(np.in1d(index0, old_index1))[0]
        if len(pts0) >= 50 and (i == 0 or len(match2d3d) >= 4):
            for k in range(len(pts0)):
                p3d_idx = cameras[i].match2d3d[index0[k]]
                if p3d_idx != -1:
                    cameras[i+j].match2d3d[index1[k]] = p3d_idx
                else:
                    point_cloud.append(Point3d(0, 0, 0, color=cameras[i].color[index0[k]]))
                    cameras[i].match2d3d[index0[k]] = len(point_cloud) - 1
                    cameras[i+j].match2d3d[index1[k]] = len(point_cloud) - 1
            cameras[i].reconstrucable = True
            cameras[i + j].reconstrucable = True
            i = i + j
            old_index1 = index1.copy()
            j = 1
        else:
            j += 1
        pbar.update(1)

new_cams = []
for cam in cameras:
    if cam.reconstrucable:
        new_cams.append(cam)
cameras = new_cams

init_idx = 0

pts0, pts1, pts3d_idx = cameras[init_idx].getMatch(cameras[init_idx+1])
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1, mask=None)       
pts0_ = pts0[mask.ravel() == 1]
pts1_ = pts1[mask.ravel() == 1]
print("ESSENTIAL MATRIX:")
print(E)
_, R, t, mask = cv2.recoverPose(E, pts0_, pts1_, K)
cameras[init_idx].setRt(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0], [0], [0]]))
cameras[init_idx + 1].setRt(R, t)
points_3d = triangulate(cameras[init_idx].getP(K), cameras[init_idx+1].getP(K), pts0, pts1)
for i in range(len(pts3d_idx)):
    if not point_cloud[pts3d_idx[i]].triangulate:
        point_cloud[pts3d_idx[i]].fromNumpy(points_3d[i])

for i in tqdm(range(2, len(cameras))):
    point_3d_idx = cameras[i].match2d3d
    mask = np.array([point_cloud[j].triangulate if j != -1 else False for j in point_3d_idx])
    points_3d_n = np.float32([point_cloud[i].numpy() for i in np.array(point_3d_idx)[mask == 1]])
    points_2d_n = np.float32([cameras[i].kp[j].pt for j in np.where(mask)[0]])
    ret, rvecs, trans, inliers = cv2.solvePnPRansac(points_3d_n, points_2d_n, K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
    Rot, _ = cv2.Rodrigues(rvecs)
    cameras[i].setRt(Rot, trans)
    pts0, pts1, pts3d_idx = cameras[i-1].getMatch(cameras[i])
    points_3d = triangulate(cameras[i-1].getP(K), cameras[i].getP(K), pts0, pts1)
    for j in range(len(pts3d_idx)):
        if not point_cloud[pts3d_idx[j]].triangulate:
            point_cloud[pts3d_idx[j]].fromNumpy(points_3d[j])

points_3d = []
colors = []
for pt in point_cloud:
    if pt.triangulate:
        coord, color = pt.numpy(), pt.color
        points_3d.append(coord)
        colors.append(color)
campos = [cam.getPos() for cam in cameras]
to_ply(path, img_dir, np.array(points_3d), np.array(colors), False, False)
to_ply(path, img_dir, np.array(campos), np.ones_like(np.array(campos))*255, False, True)
