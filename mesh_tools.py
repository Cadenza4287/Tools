import numpy as np
import cv2

def extract_v(s):
    v = []
    v_s = s.split(' ')
    for _, num in enumerate(v_s[1:]):
        v.append(float(num))

    return v

def extract_f(s):
    f = []
    f_s = s.split(' ')
    for _, num in enumerate(f_s[1:]):
        if '//' in s:
            idx = num.find('/')
            f.append(int(num[:idx]))
        else:
            f.append(int(num))

    return f

def read_obj(filename ):
    with open(filename) as f:
        vertexs = []
        faces = []
        for line in f.readlines():
            # line = line.replace('\n','').replace('\r','')
            line = " ".join(line.split())
            if len(line) > 0:
                if not '#' in line:
                    if not 'vn' in line:
                        if line[0] == 'v':
                            vertexs.append(extract_v(line))
                        elif line[0] == 'f':
                            faces.append(extract_f(line))

        return vertexs, faces


def save_obj(filename, vertexs, faces):
    with open(filename, 'w') as f:
        for i in range(len(vertexs)):
            f.write('v {:f} {:f} {:f}\n'.format(vertexs[i][0], vertexs[i][1],
                                                vertexs[i][2]))
        for i in range(len(faces)):
            f.write('f {:d} {:d} {:d}\n'.format(
                int(faces[i][0]), int(faces[i][1]), int(faces[i][2])))


def back_project(image_size, K, Rt, vertex , faces):
    pc_t = np.transpose(vertex)
    pc_t = np.vstack((pc_t, np.ones([1, np.shape(vertex)[0]])))
    pc_cam = np.dot(Rt, pc_t)

    uvs = np.dot(K, pc_cam)

    points_2d = np.floor(uvs / uvs[2, :] + 0.5)
    points_2d[points_2d < 0] = -1
    points_2d[0, :][points_2d[0, :] >= image_size[1]] = -1
    points_2d[1, :][points_2d[1, :] >= image_size[0]] = -1
    points_2d = np.transpose(points_2d[:2, :])
    points_2d = np.array(points_2d , np.int32)

    triangles = []
    for face in faces:
        triangles.append([points_2d[face[0]-1] , points_2d[face[1]-1] , points_2d[face[2]-1]])

    temp_image = np.zeros(list(image_size), dtype=np.uint8)
    for triangle in triangles:
        flag = True
        for point in triangle:
            if point.any() < 0:
                flag = False
        if flag:
            points = np.array(np.reshape(triangle, [-1, 1, 2]), np.int32)
            cv2.fillPoly(temp_image, [points], 255, 1)

    return temp_image , points_2d , uvs

def direct_norm(pc):
    pca = PCA(n_components=1)
    pca.fit(pc)
    direct = pca.components_
    theta = np.arctan2(direct[0][1] , direct[0][0])
    mat = cv2.Rodrigues(np.array([0.0 , 0.0 , theta]))[0]
    new_pc = np.dot(pc , mat)
    temp = np.array(new_pc[: , 0])
    new_pc[: , 0] = new_pc[: , 1]
    new_pc[: , 1] = - temp
    return new_pc
