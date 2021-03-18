import tensorflow as tf
import cv2
import numpy as np
from sklearn.decomposition import PCA
import binvox

def read_binvox(path):
    with open(path + ".binvox", 'rb') as f:
        m = binvox.read_as_3d_array(f).data
    return m

def volume2pc(volume, vox_sz=1.25):
    size = np.max(np.shape(volume))
    center = size // 2 - 1
    idx = np.transpose(np.where(volume))
    pc = (idx - center) * vox_sz
    return pc

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

def extract_surface(volume_solid, ksize=3):
    if len(volume_solid.get_shape()) == 4:
        volume_solid_5d = tf.expand_dims(volume_solid, -1)
    elif len(volume_solid.get_shape()) == 5:
        volume_solid_5d = volume_solid
    else:
        print("error extract surface")
        sys.exit(1)

    volume_min = -tf.nn.max_pool3d(- volume_solid_5d, [1, ksize, ksize, ksize, 1],
                                   [1, 1, 1, 1, 1], 'SAME')
    volume_surface = tf.cast(
        tf.logical_and(volume_solid_5d >= 0.5, volume_min < 0.5), tf.float32)

    volume_surface_return = tf.squeeze(volume_surface , axis= - 1)
    return volume_surface_return


def align_voxel_grid(vox):

    x_shape, y_shape, z_shape = np.shape(vox)
    voxel_out = np.zeros(np.shape(vox), dtype=np.bool)

    voxel_index = np.transpose(np.where(vox))
    max_bbox = np.max(voxel_index, axis=0)
    min_bbox = np.min(voxel_index, axis=0)
    bbox_size = max_bbox - min_bbox
    x_bbox, y_bbox, z_bbox = bbox_size
    voxel_out[x_shape // 2 - x_bbox // 2: x_shape // 2 - x_bbox // 2 + x_bbox + 1,
    y_shape // 2 - y_bbox // 2: y_shape // 2 - y_bbox // 2 + y_bbox + 1,
    z_shape // 2 - z_bbox // 2: z_shape // 2 + z_bbox // 2 + 1] = vox[min_bbox[0]:max_bbox[0] + 1, min_bbox[1]:max_bbox[1] + 1,
                                               min_bbox[2]:max_bbox[2] + 1]

    return voxel_out

def generate_indices(side):
    """ Generates meshgrid indices.
        May be helpful to cache when converting lots of shapes.
    """
    r = np.arange(0, side+2)
    id1,id2,id3 = np.meshgrid(r,r,r, indexing='ij')
    return id1, id2, id3

def encode_shapelayer(voxel, id1=None, id2=None, id3=None):

    """ Encodes a project_single_view_contour grid into a shape layer
        by projecting the enclosed shape to 6 depth maps.
        Returns the shape layer and a reconstructed shape.
        The reconstructed shape can be used to recursively encode
        complex shapes into multiple layers.
        Optional parameters id1,id2,id3 can save memory when multiple
        shapes are to be encoded. They can be constructed like this:
        r = np.arange(0,project_single_view_contour.shape[0]+2)
        id1,id2,id3 = np.meshgrid(r,r,r, indexing='ij')
    """

    side = voxel.shape[0]
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[1] == voxel.shape[2], \
        'The project_single_view_contour grid needs to be a cube. It is however %dx%dx%d.' % \
        (voxel.shape[0], voxel.shape[1], voxel.shape[2])

    if id1 is None or id2 is None or id3 is None:
        id1, id2, id3 = generate_indices(side)
        pass

    # add empty border for argmax
    # need to distinguish empty tubes
    v = np.zeros((side + 2, side + 2, side + 2), dtype=np.uint8)
    v[1:-1, 1:-1, 1:-1] = voxel

    shape_layer = np.zeros((side, side, 6), dtype=np.uint16)

    # project depth to yz-plane towards negative x
    s1 = np.argmax(v, axis=0)  # returns first occurence
    # project depth to yz-plane towards positive x
    s2 = np.argmax(v[-1::-1, :, :], axis=0)  # same, but from other side
    s2 = side + 1 - s2  # correct for added border

    # set all empty tubes to 0
    s1[s1 < 1] = side + 2
    s2[s2 > side] = 0
    shape_layer[:, :, 0] = s1[1:-1, 1:-1]
    shape_layer[:, :, 1] = s2[1:-1, 1:-1]

    # project depth to xz-plane towards negative y
    s1 = np.argmax(v, axis=1)
    # project depth to xz-plane towards positive y
    s2 = np.argmax(v[:, -1::-1, :], axis=1)
    s2 = side + 1 - s2

    s1[s1 < 1] = side + 2
    s2[s2 > side] = 0
    shape_layer[:, :, 2] = s1[1:-1, 1:-1]
    shape_layer[:, :, 3] = s2[1:-1, 1:-1]

    # project depth to xy-plane towards negative z
    s1 = np.argmax(v, axis=2)
    # project depth to xy-plane towards positive z
    s2 = np.argmax(v[:, :, -1::-1], axis=2)
    s2 = side + 1 - s2

    s1[s1 < 1] = side + 2
    s2[s2 > side] = 0
    shape_layer[:, :, 4] = s1[1:-1, 1:-1]
    shape_layer[:, :, 5] = s2[1:-1, 1:-1]

    return shape_layer

def decode_shapelayer(shape_layer, id1=None, id2=None, id3=None):
    """ Decodes a shape layer to project_single_view_contour grid.
        Optional parameters id1,id2,id3 can save memory when multiple
        shapes are to be encoded. They can be constructed like this:
        r = np.arange(0,project_single_view_contour.shape[0]+2)
        id1,id2,id3 = np.meshgrid(r,r,r)
    """

    side = shape_layer.shape[0]

    if id1 is None or id2 is None or id3 is None:
        id1,id2,id3 = generate_indices(side)
        pass

    x0 = np.expand_dims(np.pad(shape_layer[:,:,0], ((1,1), (1,1)), 'constant'), axis=0)
    x1 = np.expand_dims(np.pad(shape_layer[:,:,1], ((1,1), (1,1)), 'constant'), axis=0)
    y0 = np.expand_dims(np.pad(shape_layer[:,:,2], ((1,1), (1,1)), 'constant'), axis=1)
    y1 = np.expand_dims(np.pad(shape_layer[:,:,3], ((1,1), (1,1)), 'constant'), axis=1)
    z0 = np.expand_dims(np.pad(shape_layer[:,:,4], ((1,1), (1,1)), 'constant'), axis=2)
    z1 = np.expand_dims(np.pad(shape_layer[:,:,5], ((1,1), (1,1)), 'constant'), axis=2)

    v0 = np.logical_and(x0 <= id1, id1 <= x1)
    v1 = np.logical_and(y0 <= id2, id2 <= y1)
    v2 = np.logical_and(z0 <= id3, id3 <= z1)

    return np.logical_and(np.logical_and(v0, v1), v2)[1:-1,1:-1,1:-1]

def read_shape_layer(shape_layer_path_list):
    sls = []
    for i in range(6):
        img = cv2.imread(shape_layer_path_list[i] , -1)
        img = np.array(img, np.uint16)
        sls.append(img)

    sls = np.transpose(sls, axes=[1, 2, 0])
    voxel_grid = decode_shapelayer(sls)
    voxel_grid = np.array(voxel_grid, np.bool)

    return voxel_grid

def save_shape_layer(voxel , shape_layer_path_list):
    voxel_write = np.array(voxel, np.int)
    shape_layer = encode_shapelayer(voxel_write)
    shape_layer = np.transpose(shape_layer, axes=[2, 0, 1])
    for i in range(6):
        # cv2.imshow("A" , np.array(shape_layer[i] , np.uint8) / 256)
        # cv2.waitKey(0)
        cv2.imwrite(shape_layer_path_list[i], np.array(shape_layer[i], np.int32))

def interpolate(im, x, y, z, out_size):
    """Bilinear interploation layer.
    Args:
        im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
        It is the input volume for the transformation layer (tf.float32).
        x: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for x (tf.float32).
        y: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for y (tf.float32).
        z: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for z (tf.float32).
        out_size: A tuple representing the output size of transformation layer
        (float).
    Returns:
        A transformed tensor (tf.float32).
    """
    with tf.variable_scope('interpolate'):
        num_batch = im.get_shape().as_list()[0]
        depth = im.get_shape().as_list()[1]
        height = im.get_shape().as_list()[2]
        width = im.get_shape().as_list()[3]
        channels = im.get_shape().as_list()[4]

        x = tf.to_float(x)
        y = tf.to_float(y)
        z = tf.to_float(z)
        depth_f = tf.to_float(depth)
        height_f = tf.to_float(height)
        width_f = tf.to_float(width)
        # Number of disparity interpolated.
        out_depth = out_size[0]
        out_height = out_size[1]
        out_width = out_size[2]
        zero = tf.zeros([], dtype='int32')
        # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
        max_z = tf.to_int32(tf.shape(im)[1] - 1)
        max_y = tf.to_int32(tf.shape(im)[2] - 1)
        max_x = tf.to_int32(tf.shape(im)[3] - 1)

        # Converts scale indices from [-1, 1] to [0, width/height/depth].
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0
        z = (z + 1.0) * (depth_f) / 2.0

        x0 = tf.to_int32(tf.floor(x))
        x1 = x0 + 1
        y0 = tf.to_int32(tf.floor(y))
        y1 = y0 + 1
        z0 = tf.to_int32(tf.floor(z))
        z1 = z0 + 1

        x0_clip = tf.clip_by_value(x0, zero, max_x)
        x1_clip = tf.clip_by_value(x1, zero, max_x)
        y0_clip = tf.clip_by_value(y0, zero, max_y)
        y1_clip = tf.clip_by_value(y1, zero, max_y)
        z0_clip = tf.clip_by_value(z0, zero, max_z)
        z1_clip = tf.clip_by_value(z1, zero, max_z)
        dim3 = width
        dim2 = width * height
        dim1 = width * height * depth
        base = repeat(
            tf.range(num_batch) * dim1, out_depth * out_height * out_width)
        base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
        base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
        base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
        base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

        idx_z0_y0_x0 = base_z0_y0 + x0_clip
        idx_z0_y0_x1 = base_z0_y0 + x1_clip
        idx_z0_y1_x0 = base_z0_y1 + x0_clip
        idx_z0_y1_x1 = base_z0_y1 + x1_clip
        idx_z1_y0_x0 = base_z1_y0 + x0_clip
        idx_z1_y0_x1 = base_z1_y0 + x1_clip
        idx_z1_y1_x0 = base_z1_y1 + x0_clip
        idx_z1_y1_x1 = base_z1_y1 + x1_clip

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
        i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
        i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
        i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
        i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
        i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
        i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
        i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

        # Finally calculate interpolated values.
        x0_f = tf.to_float(x0)
        x1_f = tf.to_float(x1)
        y0_f = tf.to_float(y0)
        y1_f = tf.to_float(y1)
        z0_f = tf.to_float(z0)
        z1_f = tf.to_float(z1)
        # Check the out-of-boundary case.
        x0_valid = tf.to_float(
            tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
        x1_valid = tf.to_float(
            tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
        y0_valid = tf.to_float(
            tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
        y1_valid = tf.to_float(
            tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
        z0_valid = tf.to_float(
            tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
        z1_valid = tf.to_float(
            tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

        w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                    (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                    (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                    (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                    1)
        w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                    (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                    1)
        w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                    (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                    (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                    (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                    1)
        w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                    (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                    1)

        output = tf.add_n([
            w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
            w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
            w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
            w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
        ])
        return output

def move_voxel(voxel , bias):
    bias_x = bias[0]
    bias_y = bias[1]
    bias_z = bias[2]

    output_voxel = np.zeros_like(voxel)
    index = np.transpose(np.where(voxel))
    bbox_max = np.max(index , axis= 0)
    bbox_min = np.min(index , axis= 0)
    output_voxel[bbox_min[0] + bias_x:bbox_max[0] + bias_x ,bbox_min[1] + bias_y:bbox_max[1] + bias_y , bbox_min[2] + bias_z:bbox_max[2] + bias_z ]= \
        voxel[bbox_min[0]:bbox_max[0] , bbox_min[1]:bbox_max[1] , bbox_min[2]:bbox_max[2]]

    return output_voxel

def meshgrid(depth, height, width):
    # remove z_near and z_far (use -1.0 and 1.0)
    with tf.variable_scope('meshgrid'):
        x_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
            [depth, height, width])
        y_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
            [depth, width, height])
        y_t = tf.transpose(y_t, [0, 2, 1])
        #sample_grid = tf.tile(
        #    tf.linspace(float(z_near), float(z_far), depth), [width * height])
        #z_t = tf.reshape(sample_grid, [height, width, depth])
        z_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, depth), [width * height]),
            [depth, width, height])
        z_t = tf.transpose(z_t, [2, 0, 1])

        # z_t = 1 / z_t
        # d_t = 1 / z_t
        # x_t /= z_t
        # y_t /= z_t

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        #d_t_flat = tf.reshape(d_t, (1, -1))
        z_t_flat = tf.reshape(z_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        #grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
        grid = tf.concat([z_t_flat, y_t_flat, x_t_flat, ones], 0)
        return grid

def repeat(x, n_repeats):
    with tf.variable_scope('repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
                ])), 1), [1, 0])
        rep = tf.to_int32(rep)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

def transform(theta, scale, input_voxel, out_size):
    with tf.variable_scope('transform'):
        num_batch = input_voxel.get_shape().as_list()[0]
        num_channels = input_voxel.get_shape().as_list()[4]
        theta = tf.reshape(theta, (-1, 4, 4))
        theta = tf.cast(theta, 'float32')

        out_depth = out_size[0]
        out_height = out_size[1]
        out_width = out_size[2]
        grid = meshgrid(out_depth, out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

        # Transform A x (x_t', y_t', 1, d_t)^T -> (x_s, y_s, z_s, 1).
        t_g = tf.matmul(theta, grid)
        z_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
        x_s = tf.slice(t_g, [0, 2, 0], [-1, 1, -1])


        z_s_flat = tf.reshape(z_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        x_s_flat = tf.reshape(x_s, [-1])

        input_transformed = interpolate(input_voxel, x_s_flat, y_s_flat, z_s_flat, out_size)

        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_depth, out_height, out_width, num_channels]))

        return output

def get_transform_matrix_tf_gpu(paras):
    """Get the 4x4 Transfromation matrix. tf version."""
    rotate, translate = paras[0], paras[1]
    rot_x = rotate[0]
    rot_y = rotate[1]
    rot_z = rotate[2]

    sin_x = tf.sin(rot_x)
    cos_x = tf.cos(rot_x)
    sin_y = tf.sin(rot_y)
    cos_y = tf.cos(rot_y)
    sin_z = tf.sin(rot_z)
    cos_z = tf.cos(rot_z)
    mat_x = tf.stack([[1.0, 0.0, 0.0], [0.0, cos_x, sin_x], [0.0, -sin_x, cos_x]])
    mat_y = tf.stack([[cos_y, 0.0, -sin_y], [0.0, 1.0, 0.0], [sin_y, 0.0, cos_y]])
    mat_z = tf.stack([[cos_z, sin_z, 0.0], [-sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]])

    rotation_matrix = tf.matmul(tf.matmul(mat_z, mat_y), mat_x)
    translate = tf.reshape(translate, [4, 1])
    extrinsic_matrix = tf.concat([rotation_matrix, [[0, 0, 0]]], 0)
    extrinsic_matrix = tf.concat([extrinsic_matrix, translate], 1)
    return extrinsic_matrix