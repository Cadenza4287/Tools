import tensorflow as tf
import numpy as np

class layers:

    @staticmethod
    def batch_norm(x, name, is_training, reuse):
        return tf.contrib.layers.batch_norm(x, scope=name, is_training=is_training, reuse=reuse)

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x, label):
        if label == 'relu':
            return layers.relu(x)
        if label == 'lrelu':
            return layers.lrelu(x, leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.variable_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, output_dim, name):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            input_dim = x.get_shape()[1]
            weight = tf.get_variable('weight', [input_dim, output_dim], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            y = tf.nn.bias_add(tf.matmul(x, weight), bias)
            layers.variable_sum(weight, name)
        return y

    @staticmethod
    def maxpool2d(x, kernel, stride, name , pad='SAME'):
        kernels = [1, kernel, kernel, 1]
        strides = [1, stride, stride, 1]
        y = tf.nn.max_pool(x, ksize=kernels, strides=strides, padding=pad , name= name)
        return y

    @staticmethod
    def maxpool3d(x, kernel, stride, name , pad='SAME'):
        kernels = [1, kernel, kernel, kernel, 1]
        strides = [1, stride, stride, stride, 1]
        y = tf.nn.max_pool3d(x, ksize=kernels, strides=strides, padding=pad , name= name)
        return y

    @staticmethod
    def conv1d(x, kernel, output_dim, stride, name, pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            input_dim = x.get_shape()[2]
            weight = tf.get_variable('weight', [kernel, input_dim, output_dim], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            strides = [1, stride, 1]
            y = tf.nn.bias_add(tf.nn.conv1d(x, weight, strides[1], pad), bias, name="bias_add")
            layers.variable_sum(weight, name)
        return y

    @staticmethod
    def conv2d(x, kernel, output_dim, stride, name, pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            input_dim = x.get_shape()[3]
            weight = tf.get_variable('weight', [kernel, kernel, input_dim, output_dim], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            strides = [1, stride, stride, 1]
            y = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides, pad), bias , name= "bias_add")
            layers.variable_sum(weight, name)
        return y

    @staticmethod
    def conv3d(x, kernel, output_dim, stride, name, pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            input_dim = x.get_shape()[4]
            weight = tf.get_variable('weight', [kernel, kernel, kernel, input_dim, output_dim], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            strides = [1, stride, stride, stride, 1]
            y = tf.nn.bias_add(tf.nn.conv3d(x, weight, strides, pad), bias)
            layers.variable_sum(weight, name)
            return y

    @staticmethod
    def deconv2d(x, kernel, output_dim, stride, name, pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            [_, input_dim1, input_dim2, input_channel] = x.get_shape()
            input_dim1 = int(input_dim1)
            input_dim2 = int(input_dim2)
            input_channel = int(input_channel)
            batch_size = tf.shape(x)[0]
            weight = tf.get_variable('weight', [kernel, kernel, output_dim, input_channel], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            out_shape = [batch_size, input_dim1 * stride, input_dim2 * stride, output_dim]
            strides = [1, stride, stride, 1]
            y = tf.nn.conv2d_transpose(
                x, weight, output_shape=out_shape, strides=strides, padding=pad)
            y = tf.nn.bias_add(y, bias)
            layers.variable_sum(weight, name)
        return y

    @staticmethod
    def deconv3d(x, kernel, output_dim, stride, name, pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            [_, input_dim1, input_dim2, input_dim3, input_channels] = x.get_shape()
            input_dim1 = int(input_dim1)
            input_dim2 = int(input_dim2)
            input_dim3 = int(input_dim3)
            input_channels = int(input_channels)
            batch_size = tf.shape(x)[0]
            weight = tf.get_variable('weight', [kernel, kernel, kernel, output_dim, input_channels], initializer=xavier_init)
            bias = tf.get_variable('bias', [output_dim], initializer=zero_init)
            out_shape = [batch_size, input_dim1 * stride, input_dim2 * stride, input_dim3 * stride, output_dim]
            stride = [1, stride, stride, stride, 1]
            y = tf.nn.conv3d_transpose(
                x, weight, output_shape=out_shape, strides=stride, padding=pad)
            y = tf.nn.bias_add(y, bias)
            layers.variable_sum(weight, name)
        return y

class Losses:

    @staticmethod
    def get_corner(gt, ksize=3):
        # edge voxels
        # dilated_gt_x = tf.nn.max_pool3d(-gt, [1, 1, ksize, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        # corner voxels
        dilated_gt_x = tf.nn.max_pool3d(-gt, [1, ksize, 1, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_x = dilated_gt_x + gt
        # dilated_gt_y = tf.nn.max_pool3d(-gt, [1, ksize, 1, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        dilated_gt_y = tf.nn.max_pool3d(-gt, [1, 1, ksize, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_y = dilated_gt_y + gt
        # dilated_gt_z = tf.nn.max_pool3d(-gt, [1, ksize, ksize, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        dilated_gt_z = tf.nn.max_pool3d(-gt, [1, 1, 1, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_z = dilated_gt_z + gt
        boudary_gt = tf.reduce_min(tf.concat([boundary_gt_x, boundary_gt_y, boundary_gt_z], axis=-1), axis=-1)
        # dilated_gt = tf.nn.max_pool3d(-gt, [1, ksize, ksize, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        # boudary_gt = dilated_gt + gt
        return boudary_gt

    @staticmethod
    def get_edge(gt, ksize=3):
        # edge voxels
        dilated_gt_x = tf.nn.max_pool3d(-gt, [1, 1, ksize, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        # corner voxels
        # dilated_gt_x = tf.nn.max_pool3d(-gt, [1, ksize, 1, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_x = dilated_gt_x + gt
        dilated_gt_y = tf.nn.max_pool3d(-gt, [1, ksize, 1, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        # dilated_gt_y = tf.nn.max_pool3d(-gt, [1, 1, ksize, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_y = dilated_gt_y + gt
        dilated_gt_z = tf.nn.max_pool3d(-gt, [1, ksize, ksize, 1, 1], [1, 1, 1, 1, 1], 'SAME')
        # dilated_gt_z = tf.nn.max_pool3d(-gt, [1, 1, 1, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt_z = dilated_gt_z + gt
        boudary_gt = tf.reduce_min(tf.concat([boundary_gt_x, boundary_gt_y, boundary_gt_z], axis=-1), axis=-1)
        # dilated_gt = tf.nn.max_pool3d(-gt, [1, ksize, ksize, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        # boudary_gt = dilated_gt + gt
        return boudary_gt

    @staticmethod
    def iou(vox1, vox2):
        intersection = np.sum(np.logical_and(vox1, vox2))
        union = np.sum(np.logical_or(vox1, vox2))
        return float(intersection) / float(union)

    @staticmethod
    def iou_tf(vox1, vox2):
        intersection = tf.reduce_sum(tf.logical_and(vox1, vox2))
        union = tf.reduce_sum(tf.logical_or(vox1, vox2))
        return tf.cast(intersection, tf.float32) / tf.cast(union, tf.float32)

    @staticmethod
    def cross_entropy(pred, gt):
        ce = np.mean(gt * np.log(pred + 1e-8) - (1 - gt) * np.log(1 - pred + 1e-8))
        return ce

    @staticmethod
    def weighted_bce(pd, gt, alpha):
        """
        weighted binary cross-entropy loss.
        """
        assert pd.get_shape() == gt.get_shape(), "pd and gt must have same shape!"
        bat = pd.get_shape()[0].value
        pd = tf.reshape(pd, [bat, -1])
        gt = tf.reshape(gt, [bat, -1])
        loss = tf.reduce_mean(- alpha * gt * tf.log(pd + 1e-8)
                              - (1 - alpha) * (1 - gt) * tf.log(1 - pd + 1e-8))
        return loss

    @staticmethod
    def adaptive_bce(pd, gt):
        """
        adaptive binary cross-entropy loss.
        """
        assert pd.get_shape() == gt.get_shape(), "pd and gt must have same shape!"
        bat = pd.get_shape()[0].value
        pd = tf.reshape(pd, [bat, -1])
        gt = tf.reshape(gt, [bat, -1])
        one_nums = tf.reduce_sum(gt, axis=1)
        all_nums = float(32 * 32 * 32)
        weight_ones = one_nums / all_nums
        weight_zeros = 1 - weight_ones
        loss = tf.reduce_mean(- weight_zeros * tf.reduce_mean(gt * tf.log(pd + 1e-8), 1)
                              - weight_ones * tf.reduce_mean((1 - gt) * tf.log(1 - pd + 1e-8)))
        return loss

    @staticmethod
    def boundary_loss(pd, gt, ksize):
        """
        boundary loss.
        """
        # assert pd.get_shape() == gt.get_shape(), "pd and gt must have same shape!"
        # extract boundary voxels
        dilated_gt = tf.nn.max_pool3d(-gt, [1, ksize, ksize, ksize, 1], [1, 1, 1, 1, 1], 'SAME')
        boundary_gt = dilated_gt + gt
        bat = pd.get_shape()[0].value
        pd = tf.reshape(pd, [bat, -1])
        gt = tf.reshape(gt, [bat, -1])
        boundary_gt = tf.reshape(boundary_gt, [bat, -1])
        boundary_idxes = tf.where(boundary_gt >= 0.5)
        boundary_gt_voxels = tf.gather(gt, boundary_idxes)
        boundary_pd_voxels = tf.gather(pd, boundary_idxes)
        loss = tf.reduce_mean(-gt * tf.log(pd + 1e-8))

        return loss

    @staticmethod
    def corner_loss(pd, gt, ksize):
        """
        corner loss.
        corner is intersection of x y z 1D conv results.
        """
        # assert pd.get_shape() == gt.get_shape(), "pd and gt must have same shape!"
        # extract corner voxels
        boundary_gt = Losses.get_corner(gt, ksize)
        bat = pd.get_shape()[0].value
        pd = tf.reshape(pd, [bat, -1])
        gt = tf.reshape(gt, [bat, -1])
        boundary_gt = tf.reshape(boundary_gt, [bat, -1])
        boundary_idxes = tf.where(boundary_gt >= 0.5)
        boundary_gt_voxels = tf.gather(gt, boundary_idxes)
        boundary_pd_voxels = tf.gather(pd, boundary_idxes)
        loss = tf.reduce_mean(-gt * tf.log(pd + 1e-8))

        return loss

    @staticmethod
    def edge_loss(pd, gt, ksize):
        """
        corner loss.
        corner is intersection of x y z 1D conv results.
        """
        # assert pd.get_shape() == gt.get_shape(), "pd and gt must have same shape!"
        # extract corner voxels
        boundary_gt = Losses.get_edge(gt, ksize)
        bat = pd.get_shape()[0].value
        pd = tf.reshape(pd, [bat, -1])
        gt = tf.reshape(gt, [bat, -1])
        boundary_gt = tf.reshape(boundary_gt, [bat, -1])
        boundary_idxes = tf.where(boundary_gt >= 0.5)
        boundary_gt_voxels = tf.gather(gt, boundary_idxes)
        boundary_pd_voxels = tf.gather(pd, boundary_idxes)
        loss = tf.reduce_mean(-gt * tf.log(pd + 1e-8))

        return loss

    @staticmethod
    def chamfer_distance_single(inputs):
        """
            suqared distance, to be same with chamfer distance in P2M
        """
        pc1 = inputs[0]
        pc2 = inputs[1]
        num_p1 = tf.shape(pc1)[0]
        num_p2 = tf.shape(pc2)[0]
        num_f1 = tf.shape(pc1)[1]
        num_f2 = tf.shape(pc2)[1]
        # specify the number of pc1 and pc2
        # num_p1= pc1.get_shape()[0]
        # num_p2= pc2.get_shape()[0]
        # num_f1= pc1.get_shape()[1]
        # num_f2= pc2.get_shape()[1]
        exp_pc1 = tf.tile(pc1, (num_p2, 1))  # [num_p1 * num_p2, num_f1]
        exp_pc1 = tf.reshape(exp_pc1, [num_p2, num_p1, num_f1])  # [num_p2, num_p1, num_f1]
        exp_pc2 = tf.reshape(pc2, [num_p2, 1, num_f2])  # [num_p2, 1, num_f1]
        # exp_pc2 = tf.tile(exp_pc2, (1, num_p1, 1)) # [num_p2, num_p1, num_f1], use broadcasting automatically, no need to used tile
        distance_matrix = tf.squared_difference(exp_pc1, exp_pc2)  # [num_p2, num_p1]
        distance_matrix = tf.reduce_sum(distance_matrix, axis=2)  # [num_p2, num_p1]
        d1_2_all = tf.reduce_min(distance_matrix, axis=0)  # [num_p1]
        d2_1_all = tf.reduce_min(distance_matrix, axis=1)  # [num_p2]
        idx1 = tf.arg_min(distance_matrix, 0, tf.int32)  # [num_p1]
        idx2 = tf.arg_min(distance_matrix, 1, tf.int32)  # [num_p2]
        return [d1_2_all, idx1, d2_1_all, idx2]

    @staticmethod
    def chamfer_distance(pc1, pc2):
        """
        chamfer distance with tensorflow.\n
        input:\n
        pc1: point cloud 1 [batch_size, point_number, feature_number]\n
        pc2: point cloud 2 [batch_size, point_number, feature_number]\n
        return:\n
        d1_2: distance from pc1 to pc2\n
        d2_1: distance from pc2 to pc1\n
        idx1: indexs in pc2 correspond to pc1
        idx2: indexs in pc1 correspond to pc2
        """
        shape1 = pc1.get_shape()
        shape2 = pc2.get_shape()
        # assert shape1[0] == shape2[0], "batch size must be same!"
        assert shape1[2] == shape2[2], "feature number must be same!"
        num_p1 = shape1[1]
        num_p2 = shape2[1]
        output_type = [tf.float32, tf.int32, tf.float32, tf.int32]
        res = tf.map_fn(Losses.chamfer_distance_single, elems=(pc1, pc2), dtype=output_type)
        return res


    @staticmethod
    def edge_distance(pd, edge_index):
        index_0 = edge_index[:, 0]
        index_1 = edge_index[:, 1]
        points_0 = tf.gather(pd, index_0, axis=1)
        points_1 = tf.gather(pd, index_1, axis=1)
        edges = points_0 - points_1
        edge_length = tf.reduce_sum(tf.square(edges), -1)

        avg_length = tf.reduce_mean(tf.reduce_mean(edge_length, 1))

        return avg_length

    @staticmethod
    def unit(vec):
        return tf.nn.l2_normalize(vec, dim=-1)

    @staticmethod
    def point_normal_loss(pd, edge_index, gt_normals, dis_index, batch_index):
        # print(dis_index.get_shape(), batch_index.get_shape())
        bat, point_num = int(pd.get_shape()[0]), int(pd.get_shape()[1])
        index_0 = edge_index[:, 0]
        index_1 = edge_index[:, 1]
        points_0 = tf.gather(pd, index_0, axis=1)
        points_1 = tf.gather(pd, index_1, axis=1)
        # concat dis_index and batch_index
        index_norm = tf.stack([batch_index, dis_index], axis=-1)
        # print(index_norm.get_shape())
        index_norm = tf.reshape(index_norm, (bat * point_num, 2))
        # print(index_norm.get_shape())
        # print(gt_normals.get_shape())
        edges = points_0 - points_1
        normals = tf.gather_nd(gt_normals, index_norm)
        # print(normals.get_shape())
        normals = tf.reshape(normals, (bat, point_num, 3))
        # print(normals.get_shape())
        normals_gather = tf.gather(normals, index_0, axis=1)

        cosine = tf.abs(tf.reduce_sum(tf.multiply(Losses.unit(points_0), Losses.unit(normals_gather)), axis=-1))
        normal_loss = tf.reduce_mean(tf.reduce_mean(cosine, axis=1))

        return normal_loss

    @staticmethod
    def face_normals(pd, face_index):
        """compute face normals"""
        point_0 = tf.gather(pd, face_index[:, 0], axis=1)
        point_1 = tf.gather(pd, face_index[:, 1], axis=1)
        point_2 = tf.gather(pd, face_index[:, 2], axis=1)
        edge_0 = Losses.unit(point_1 - point_0)
        edge_1 = Losses.unit(point_2 - point_1)
        normals = tf.cross(edge_0, edge_1)

        return normals

    @staticmethod
    def face_normal_loss(pd, face_index, face_adjacency):
        """
        compute face intersection angle.
        want them to be zeros (flat). cosine is 1
        """
        normals = losses.face_normals(pd, face_index)
        # print(normals.get_shape())
        # print(face_index.get_shape())
        # print(face_adjacency.get_shape())
        normals_0 = tf.gather(normals, face_adjacency[:, 0], axis=1)
        normals_1 = tf.gather(normals, face_adjacency[:, 1], axis=1)

        cosine = tf.reduce_sum(tf.multiply(normals_0, normals_1), axis=-1)
        normal_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(cosine - 1), axis=1))

        return normal_loss


    @staticmethod
    def face_projection_loss(pd_proj, gt_proj):
        """
        compute projection loss (L2)
        """
        bat = int(pd_proj.get_shape()[0])
        pd_proj = tf.reshape(pd_proj, shape=(bat, -1))
        gt_proj = tf.reshape(gt_proj, shape=(bat, -1))
        dif = tf.abs(pd_proj, gt_proj)
        proj_loss = tf.reduce_mean(tf.square(dif), axis=1)
        return tf.reduce_mean(proj_loss)
