import os
import cv2
import re
import numpy as np
from tqdm import tqdm
from tinydb import TinyDB, Query
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

# path = os.getcwd()
path = '..'
img_dir = path + '/gustav/'
images = os.listdir(img_dir)
images = sorted( filter( lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir) ) )
downscale = 2
convert_gray = False
cameras = []
point_cloud = []

class Point3d:
    def __init__(self, x, y, z, color=np.array([0,0,0])):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
    
    def fromNumpy(self, pt):
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
    
    def numpy(self):
        return np.array([self.x, self.y, self.z])

class Camera:
    def __init__(self, id, img, kp, desc, match2d3d):
        self.id = id
        self.img = img
        self.kp = kp
        self.desc = desc 
        self.match2d3d = match2d3d
        self.Rt = None
        self.reconstrucable = False
        self.reconstruct = False

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True
    
    def getRt(self):
        return self.Rt[:3,:3], self.Rt[:3, 3]
    
    def getP(self, K):
        return np.matmul(K, self.Rt)
    
    def getPos(self):
        pts = np.array([[0,0,0]]).T
        pts = self.Rt[:3,:3].T.dot(pts)- self.Rt[:3,3][:,np.newaxis]
        return pts[:,0]
    
    def getFeature(self):
        return (self.kp, self.desc)

def get_camera_intrinsic_params(images_dir, downscale):
    K = []
    img = Image.open(images_dir + os.listdir(images_dir)[1])
    exif = {
        'FocalLength': 26,
        'Make': "notindatabase",
        'Model': "notindatabase",
        'ExifImageWidth': img.size[0],
        'ExifImageHeight': img.size[1],
        'SensorWidth': 35
    }
    exifdata = img.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        if tag in exif.keys():
            print(f"{tag:25}: {exifdata.get(tag_id)}")
            exif[tag] = exifdata.get(tag_id)
    Camera = Query()
    db = TinyDB('cameras.json')
    res = db.search(Camera.make.search(exif['Make'].split(' ')[0], flags=re.IGNORECASE) & 
                    Camera.model.search(exif['Model'].strip(), flags=re.IGNORECASE))
    exif['SensorWidth'] = res[0]['sensor_width'] if len(res) > 0 else exif['SensorWidth']
    exif['FocalLength'] = (exif['FocalLength'][0]/exif['FocalLength'][1]) if isinstance(exif['FocalLength'], tuple) else exif['FocalLength']
    image_width = exif['ExifImageWidth'] if exif['ExifImageWidth'] > exif['ExifImageHeight'] else exif['ExifImageHeight']
    focal_length = (exif['FocalLength']/exif['SensorWidth'])*image_width
    K.append([focal_length / float(downscale), 0, exif['ExifImageWidth']/(2 * float(downscale))])
    K.append([0, focal_length / float(downscale), exif['ExifImageHeight']/(2 * float(downscale))])
    K.append([0, 0, 1])
    return exif, np.array(K, dtype=float)

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
    descript = detect
    kp = detect.detect(imggray, None)
    kp,des = descript.compute(imggray, kp)
    return kp,des

def match_feature(fea0, fea1):
    kp0, des0 = fea0.getFeature()
    kp1, des1 = fea1.getFeature()

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    index0 = np.int32([m.queryIdx for m in good])
    index1 = np.int32([m.trainIdx for m in good])

    return pts0, pts1, index0, index1

def featureTracking(image_ref, image_cur, px_ref):
    lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2

def triangulate(cam1, cam2, idx0, idx1, K):
    points_3d = cv2.triangulatePoints(cam1.getP(K), cam2.getP(K), np.float32([cam1.kp[i].pt for i in idx0]).T, np.float32([cam2.kp[i].pt for i in idx1]).T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    for w, i in enumerate(idx0):
        if cam1.match2d3d[i] != -1:
            cam2.match2d3d[idx1[w]] = cam1.match2d3d[i]
        else:
            pt3d =  Point3d(0,0,0)
            pt3d.fromNumpy(points_3d[w])
            point_cloud.append(pt3d)
            cam2.match2d3d[idx1[w]] = len(point_cloud) - 1
            cam1.match2d3d[i] = len(point_cloud) - 1

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

exif, K = get_camera_intrinsic_params(img_dir, downscale)
# K = np.array([[718.8560/downscale, 0, 607.1928/downscale], [0, 718.8560/downscale, 185.2157/downscale], [0,0,1]])

prev_idx1 = np.array([])
j = 0
for i in tqdm(range(len(images))):
    if images[i].split('.')[-1] in ['JPG', 'jpg', 'PNG', 'png', 'raw']:
        img = cv2.imread(img_dir + images[i])
        if img.shape[0] == exif['ExifImageWidth'] and img.shape[1] == exif['ExifImageHeight']:
            img = img_downscale(img, downscale)
            kp, des = extract_features(img, convert_gray)
            cameras.append(Camera(images[i], img, kp, des, np.ones((len(kp),), dtype='int32')*-1))
            if j > 0:
                pts0_, pts1_, idx0, idx1 = match_feature(cameras[j-1], cameras[j])
                E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
                idx0, idx1 = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1]
                if j == 1:
                    _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)
                    cameras[j-1].setRt(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0], [0], [0]]))
                else:
                    match = np.where(np.in1d(idx0, prev_idx1))[0]
                    if len(match) < 8: continue
                    ret, rvecs, t, inliers = cv2.solvePnPRansac(np.float32([point_cloud[cameras[j-1].match2d3d[idx0[m]]].numpy() for m in match]), np.float32([cameras[j].kp[idx1[m]].pt for m in match]), K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvecs)
                cameras[j].setRt(R, t)
                prev_idx1 = idx1.copy()
                triangulate(cameras[j-1], cameras[j], idx0, idx1, K)
            j += 1
            for k in range(len(cameras[-1].kp)):
                img = cv2.circle(img, (int(cameras[-1].kp[k].pt[0]), int(cameras[-1].kp[k].pt[1])), 2, (0, 0, 255), 2)
            cv2.imshow('vid', img)
            cv2.waitKey(1)

to_ply(img_dir, np.array([pt.numpy() for pt in point_cloud]), np.array([pt.color for pt in point_cloud]))
to_ply(img_dir, np.array([cam.getPos() for cam in cameras]), np.ones_like(np.array([cam.getPos() for cam in cameras]))*255, '_campos.ply')