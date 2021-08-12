import os
import cv2
import numpy as np
from tqdm import tqdm

from PIL import Image
from PIL.ExifTags import TAGS
import math
from tinydb import TinyDB, Query
import re

from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

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

def focalsFromHomography(H):

    if H.shape[0] != 3 or H.shape[1] != 3:
        return None, None, False, False
    
    h = H.flatten()

    d1 = 0
    d2 = 0
    v1 = 0 
    v2 = 0

    f1_ok = True
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2
    if v1 < v2:
        v1, v2 = v2, v1
    if  v1 > 0 and v2 > 0:
         f1 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0: 
        f1 = math.sqrt(v1)
    else: 
        f1_ok = False

    f0_ok = True
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
    v1 = -h[2] * h[5] / d1
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        f0 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0:
        f0 = math.sqrt(v1)
    else:
        f0_ok = False
    
    return f0, f1, f0_ok, f1_ok 

def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img

def extract_features(imggray):
    # imggray = cv2.cvtColor(imggray, cv2.COLOR_BGR2GRAY)
    detect = cv2.SIFT_create()
    kp = detect.detect(imggray, None)
    kp,des = detect.compute(imggray, kp)
    return kp,des

def match_feature(fea0, fea1):
    kp0, des0 = fea0
    kp1, des1 = fea1

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

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

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

def camera_pos(R,t,scale=.5,depth=.5):
    C = -t #camera center (in world coordinate system)
    #generating 5 corners of camera polygon 
    pt1 = np.array([[0,0,0]]).T #camera centre
    pt2 = np.array([[scale,-scale,depth]]).T #upper right 
    pt3 = np.array([[scale,scale,depth]]).T #lower right 
    pt4 = np.array([[-scale,-scale,depth]]).T #upper left
    pt5 = np.array([[-scale,scale,depth]]).T #lower left
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)
    pts = pt1
    #Transforming to world-coordinate system
    pts = R.T.dot(pts)+C[:,np.newaxis]
    #Generating a list of vertices to be connected in polygon
    # verts = [ [ pts[:,0],pts[:,1],pts[:,2] ], [ pts[:,0],pts[:,2],pts[:,-1] ],
    #           [ pts[:,0],pts[:,-1],pts[:,-2] ], [ pts[:,0],pts[:,-2],pts[:,1] ] ]
    return pts[:,0]

print('--------------------------------------------------------------------------------')
print('CAMERA INFO')

path = os.getcwd()
img_dir = path + '/data/cat/'
downscale = 2
densify = False
bundle_adjustment = False
images, K = get_camera_intrinsic_params(img_dir)
# K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
# K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])
# K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
# images = images[50:100]
# images = ['DSC_0226.JPG', 'DSC_0230.JPG', 'DSC_0214.JPG', 'DSC_0193.JPG', 'DSC_0287.JPG', 'DSC_0303.JPG', 'DSC_0283.JPG', 'DSC_0276.JPG', 'DSC_0304.JPG', 'DSC_0144.JPG', 'DSC_0229.JPG', 'DSC_0205.JPG', 'DSC_0148.JPG', 'DSC_0138.JPG', 'DSC_0224.JPG', 'DSC_0164.JPG', 'DSC_0298.JPG', 'DSC_0157.JPG', 'DSC_0233.JPG', 'DSC_0141.JPG', 'DSC_0258.JPG', 'DSC_0253.JPG', 'DSC_0250.JPG', 'DSC_0171.JPG', 'DSC_0239.JPG', 'DSC_0158.JPG', 'DSC_0237.JPG', 'DSC_0268.JPG', 'DSC_0295.JPG', 'DSC_0156.JPG', 'DSC_0235.JPG', 'DSC_0180.JPG', 'DSC_0200.JPG', 'DSC_0161.JPG', 'DSC_0223.JPG', 'DSC_0245.JPG', 'DSC_0254.JPG', 'DSC_0234.JPG', 'DSC_0163.JPG', 'DSC_0249.JPG', 'DSC_0204.JPG', 'DSC_0217.JPG', 'DSC_0281.JPG', 'DSC_0241.JPG', 'DSC_0273.JPG', 'DSC_0307.JPG', 'DSC_0185.JPG', 'DSC_0286.JPG', 'DSC_0294.JPG', 'DSC_0300.JPG', 'DSC_0284.JPG', 'DSC_0291.JPG', 'DSC_0279.JPG', 'DSC_0225.JPG', 'DSC_0285.JPG', 'DSC_0305.JPG', 'DSC_0203.JPG', 'DSC_0202.JPG', 'DSC_0186.JPG', 'DSC_0176.JPG', 'DSC_0301.JPG', 'DSC_0290.JPG', 'DSC_0288.JPG', 'DSC_0175.JPG', 'DSC_0196.JPG', 'DSC_0136.JPG', 'DSC_0189.JPG', 'DSC_0166.JPG', 'DSC_0251.JPG', 'DSC_0173.JPG', 'DSC_0228.JPG', 'DSC_0263.JPG', 'DSC_0243.JPG', 'DSC_0270.JPG', 'DSC_0172.JPG', 'DSC_0192.JPG', 'DSC_0240.JPG', 
# 'DSC_0170.JPG', 'DSC_0145.JPG', 'DSC_0236.JPG', 'DSC_0289.JPG', 'DSC_0181.JPG', 'DSC_0220.JPG', 'DSC_0212.JPG', 'DSC_0277.JPG', 'DSC_0271.JPG', 'DSC_0238.JPG', 'DSC_0188.JPG', 'DSC_0232.JPG', 'DSC_0278.JPG', 'DSC_0262.JPG', 'DSC_0184.JPG', 'DSC_0139.JPG', 'DSC_0282.JPG', 'DSC_0257.JPG', 'DSC_0222.JPG', 'DSC_0274.JPG', 'DSC_0187.JPG', 'DSC_0155.JPG', 'DSC_0218.JPG', 'DSC_0177.JPG', 'DSC_0147.JPG', 'DSC_0267.JPG', 'DSC_0154.JPG', 'DSC_0169.JPG', 'DSC_0264.JPG', 'DSC_0151.JPG', 'DSC_0209.JPG', 'DSC_0297.JPG', 'DSC_0150.JPG', 'DSC_0255.JPG', 'DSC_0280.JPG', 'DSC_0198.JPG', 'DSC_0160.JPG', 'DSC_0246.JPG', 'DSC_0256.JPG', 'DSC_0265.JPG', 'DSC_0179.JPG', 'DSC_0152.JPG', 'DSC_0199.JPG', 'DSC_0259.JPG', 'DSC_0215.JPG', 'DSC_0306.JPG', 'DSC_0296.JPG', 'DSC_0207.JPG', 'DSC_0137.JPG', 'DSC_0195.JPG', 'DSC_0211.JPG', 'DSC_0266.JPG', 'DSC_0299.JPG', 'DSC_0191.JPG', 'DSC_0197.JPG', 'DSC_0252.JPG', 'DSC_0242.JPG', 'DSC_0206.JPG', 'DSC_0308.JPG', 'DSC_0208.JPG', 'DSC_0183.JPG', 'DSC_0143.JPG', 'DSC_0260.JPG', 'DSC_0231.JPG', 'DSC_0140.JPG', 'DSC_0216.JPG', 'DSC_0159.JPG', 'DSC_0292.JPG', 'DSC_0244.JPG', 'DSC_0247.JPG', 'DSC_0178.JPG', 'DSC_0302.JPG', 'DSC_0146.JPG', 'DSC_0261.JPG', 'DSC_0248.JPG', 'DSC_0227.JPG', 'DSC_0182.JPG', 'DSC_0167.JPG', 'DSC_0194.JPG', 'DSC_0309.JPG', 'DSC_0168.JPG', 'DSC_0310.JPG', 'DSC_0190.JPG', 'DSC_0269.JPG', 'DSC_0221.JPG', 'DSC_0149.JPG', 'DSC_0174.JPG', 'DSC_0165.JPG', 'DSC_0293.JPG', 'DSC_0210.JPG', 'DSC_0275.JPG', 'DSC_0201.JPG', 'DSC_0162.JPG', 'DSC_0153.JPG', 'DSC_0142.JPG', 'DSC_0213.JPG', 'DSC_0219.JPG', 'DSC_0272.JPG']
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
R_t_1 = np.empty((3, 4))
P1 = np.matmul(K, R_t_0)
P2 = np.empty((3, 4))
Xtot = np.zeros((0, 3))
colorstot = np.zeros((0, 3))
campos = np.zeros((0,3))

print('--------------------------------------------------------------------------------')
print('INTRINSIC MATRIX:')
print(K)
print("INPUT PATH:")
print(img_dir)
print("NUMBER OF IMAGE:")
print(len(images))

print('--------------------------------------------------------------------------------')
print('EXTRACT FEATURE')

all_feature = []
for i in tqdm(range(len(images))):
    img = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
    kp, des = extract_features(img)
    all_feature.append((kp, des))

print('--------------------------------------------------------------------------------')
print('MATCH FEATURE')

min_le = 1000
ale = [0]
all_pts = []
old_index1 = np.array([])
i = 0
j = 1
with tqdm(total=len(all_feature)-1) as pbar:
    while i != len(all_feature) - 1 and i + j < len(all_feature):
        pts0, pts1, index0, index1 = match_feature(all_feature[i], all_feature[i + j])
        match2d3d = np.where(np.in1d(index0, old_index1))[0]
        if len(pts0) >= 50 and (i == 0 or len(match2d3d) >= 4):
            i = i + j
            ale.append(i)
            all_pts.append((pts0, pts1, index0, index1))
            old_index1 = index1.copy()
            j = 1
        else:
            j += 1
        pbar.update(1)
print('NUMBER OF RECONSTRUCABLE IMAGE:')
print(len(ale))

print('--------------------------------------------------------------------------------')
print('INITIAL POINT CLOUD')

pts0, pts1, index0, index1 = all_pts[0]

E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1, mask=None)       
pts0 = pts0[mask.ravel() == 1]
pts1 = pts1[mask.ravel() == 1]
index0 = index0[mask.ravel() == 1]
index1 = index1[mask.ravel() == 1]

print("ESSENTIAL MATRIX:")
print(E)

_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
pts0 = pts0[mask.ravel() > 0]
pts1 = pts1[mask.ravel() > 0]
index0 = index0[mask.ravel() > 0]
index1 = index1[mask.ravel() > 0]

R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())
P2 = np.matmul(K, R_t_1)

pts_old = (pts0, pts1, index0, index1)
print("ROTATION TRANSLATION MATRIX:")
print(R_t_1)

cam_params = np.concatenate((cv2.Rodrigues(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32'))[0].flatten(), np.concatenate((np.array([0,0,0]), np.array([K[0][0],0,0])))))
posearr = np.array([cam_params.copy()])
cam_params = np.concatenate((cv2.Rodrigues(R)[0].flatten(), np.concatenate((t.flatten().copy(), np.array([K[0][0],0,0])))))
posearr = np.concatenate((posearr, np.array([cam_params.copy()])))
points_2d = np.array(pts0)
points_2d = np.concatenate((points_2d, np.array(pts1)))
camera_indices = np.ones((pts0.shape[0],), dtype='int32')*0
camera_indices = np.concatenate((camera_indices, np.ones((pts1.shape[0],), dtype='int32')*1))
point_indices = np.array([], dtype='int32')
campos = np.array([camera_pos(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32'), np.array([0,0,0])).ravel()])
campos = np.vstack((campos, np.array([camera_pos(R, t).ravel()])))

print('--------------------------------------------------------------------------------')
print('ADD MORE VIEW')

for i in tqdm(range(1, len(all_pts)-1)):
    # Get new infomation of new view
    current_img = img_downscale(cv2.imread(img_dir + images[ale[i+1]]), downscale)
    pts_new = all_pts[i]

    # Triangulate all match feature point of old pair
    points_3d = cv2.triangulatePoints(P1, P2, pts_old[0].T, pts_old[1].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]

    if i == 1:
        point_indices = np.concatenate((point_indices, np.arange(0, pts_old[0].shape[0]) + Xtot.shape[0]))
        point_indices = np.concatenate((point_indices, np.arange(0, pts_old[1].shape[0]) + Xtot.shape[0]))
        Xtot = points_3d.copy()
        ini_img = img_downscale(cv2.imread(img_dir + images[0]), downscale)
        colors = np.array([ini_img[l[1], l[0]] for l in np.int32(pts_old[0])])
        colorstot = np.vstack((colorstot, colors))

    # Perspective n Point (PnP)
    match2d3d = np.int32(np.where(np.in1d(pts_old[3], pts_new[2])))[0]
    match3d2d = np.int32([np.where(pts_new[2] == pts_old[3][m])[0][0] for m in match2d3d])
    notmatch3d2d = np.ones_like(pts_new[0])
    notmatch3d2d[match3d2d] = 0
    notmatch3d2d = np.where(notmatch3d2d)[0]
    points_3d_n = points_3d[match2d3d]
    points_2d_n = pts_new[1][match3d2d]

    ret, rvecs, trans, inliers = cv2.solvePnPRansac(points_3d_n, points_2d_n, K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
    Rot, _ = cv2.Rodrigues(rvecs)
    Rtnew = np.hstack((Rot, trans))
    Pnew = np.matmul(K, Rtnew)

    # Triangulate new match feature point of new image that not already triangulate by the correspondant
    points_3d = cv2.triangulatePoints(P2, Pnew, pts_new[0][notmatch3d2d].T, pts_new[1][notmatch3d2d].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]

    # Save result and change the variation for new iteration
    cam_params = np.concatenate((rvecs.copy().flatten(), np.concatenate((trans.copy().flatten(), np.array([K[0][0],0,0])))))
    posearr = np.concatenate((posearr, np.array([cam_params.copy()])))
    
    points_2d = np.concatenate((points_2d, np.array(pts_new[0][notmatch3d2d])))
    points_2d = np.concatenate((points_2d, np.array(pts_new[1][notmatch3d2d])))
    camera_indices = np.concatenate((camera_indices, np.zeros((pts_new[0][notmatch3d2d].shape[0],), dtype='int32') + i))
    camera_indices = np.concatenate((camera_indices, np.zeros((pts_new[1][notmatch3d2d].shape[0],), dtype='int32') + i+1))
    point_indices = np.concatenate((point_indices, np.arange(0, pts_new[0][notmatch3d2d].shape[0]) + Xtot.shape[0]))
    point_indices = np.concatenate((point_indices, np.arange(0, pts_new[1][notmatch3d2d].shape[0]) + Xtot.shape[0]))
    campos = np.vstack((campos, np.array([camera_pos(Rot, trans).ravel()])))

    Xtot = np.vstack((Xtot, points_3d))
    colors = np.array([current_img[l[1], l[0]] for l in np.int32(pts_new[1])[notmatch3d2d]])
    colorstot = np.vstack((colorstot, colors))

    pts_old = pts_new
    P1 = np.copy(P2)
    P2 = np.copy(Pnew)

if bundle_adjustment:
    print('--------------------------------------------------------------------------------')
    print("BUNDLE ADJUSTMENT")

    n_cameras = posearr.shape[0]
    n_points = Xtot.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((posearr.ravel(), Xtot.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    params = res.x
    posearr = params[:n_cameras * 9].reshape((n_cameras, 9))
    Xtot = params[n_cameras * 9:].reshape((n_points, 3))

print('--------------------------------------------------------------------------------')
print("Processing Point Cloud...")
print(Xtot.shape, colorstot.shape)
to_ply(path, img_dir, Xtot, colorstot, densify, False)
to_ply(path, img_dir, campos, np.ones_like(campos)*255, densify, True)
print("Done!")
