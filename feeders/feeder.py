import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from feeders import tools

import json
import matplotlib.pyplot as plt

class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):

        # NTU DATASET
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                # self.sample_name, self.label = pickle.load(f)
                self.label, self.sample_name = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                # self.sample_name, self.label = pickle.load(f, encoding='latin1')
                self.label, self.sample_name, = pickle.load(f, encoding='latin1')

        # robot dataset

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path, allow_pickle=True)

        # f = open(self.label_path)
        # self.label = json.load(f)
        #
        # ##### change data to format (N C T V M)  ######
        # self.data = np.vstack(self.data)
        # self.data = np.expand_dims(self.data, axis=4)
        # self.data = np.transpose(self.data, (0, 3, 1, 2, 4))
        #
        # ##### build (sample_name, label) tuple ######
        # final_sample_name = []
        # final_label = []
        # for _, l_item in enumerate(self.label):
        #     final_sample_name = final_sample_name + l_item['sample_name']
        #     final_label = final_label + l_item['id_action']
        # self.label = final_label
        # self.sample_name = final_sample_name

        if self.debug:
            self.label = self.label[0:20]
            self.data = self.data[0:20]
            self.sample_name = self.sample_name[0:20]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def align_frames(self, data):
        MAX_FRAME = 300
        count_overcome_max_frame = 0
        aligned = True

        # for action_idx, action_item in enumerate(data):
        for clip_idx, clip_item in enumerate(data):
            data[clip_idx] = np.array(clip_item)
            num_frame = data[clip_idx].shape[0]
            if num_frame > MAX_FRAME:
                count_overcome_max_frame = count_overcome_max_frame + 1
                # print("action_idx: {}, clip_idx: {}, size > 300".format(action_idx, clip_idx))
                data[clip_idx] = data[clip_idx][0:MAX_FRAME]
                continue
            elif MAX_FRAME % num_frame == 0:
                num_repeat = int(MAX_FRAME / num_frame)
                data[clip_idx] = np.tile(data[clip_idx], (num_repeat, 1, 1))
            elif int(MAX_FRAME / num_frame) == 1:
                # e.g. 226
                data[clip_idx] = np.vstack(
                    (data[clip_idx], data[clip_idx][0:MAX_FRAME - num_frame]))
            else:
                # e.g. 17
                num_repeat = int(MAX_FRAME / num_frame)
                padding = MAX_FRAME % num_frame
                data[clip_idx] = np.tile(data[clip_idx], (num_repeat, 1, 1))
                data[clip_idx] = np.vstack(
                    (data[clip_idx], data[clip_idx][0:padding]))

            if data[clip_idx].shape != (300, 17, 2):
                aligned = False
                print("clip_idx: {} is not correct aligned with {}".format(clip_idx, MAX_FRAME))

        if aligned == True:
            print("All the clips are aligned with size")
        else:
            print("There are some clips with shape incorrect")

        return np.array(data)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        # data_numpy = self.align_frames(data)
        data_numpy = data

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        # sample_name = loader.dataset.sample_name
        # sample_id = [name.split('.')[0] for name in sample_name]
        # index = sample_id.index(vid)
        # data, label, index = loader.dataset[index]
        # data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        data = loader.dataset[0]

        # TEST
        # plot_pose(data, 4)

        # ORIG
        data = list(data)
        N, T, V, C = data[0].shape
        M = 1

        # DON'T work 'reshape'
        # data[0] = data[0].reshape(N, C, T, V)
        # data[0] = data[0].transpose(0, 3, 1, 2)
        # Plot figure use my function
        plot_pose(data, 0)

        # Don'use author's method to plot
        # plt.ion()
        # fig = plt.figure()
        # if is_3d:
        #     from mpl_toolkits.mplot3d import Axes3D
        #     ax = fig.add_subplot(111, projection='3d')
        # else:
        #     ax = fig.add_subplot(111)
        #
        # if graph is None:
        #     p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
        #     pose = [
        #         ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
        #     ]
        #     ax.axis([-1, 1, -1, 1])
        #     for t in range(T):
        #         for m in range(M):
        #             pose[m].set_xdata(data[0, 0, t, :, m])
        #             pose[m].set_ydata(data[0, 1, t, :, m])
        #         fig.canvas.draw()
        #         plt.pause(0.001)
        # else:
        #     p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
        #     import sys
        #     from os import path
        #     sys.path.append(
        #         path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        #     G = import_class(graph)()
        #     edge = G.inward
        #     pose = []
        #     for m in range(M):
        #         a = []
        #         for i in range(len(edge)):
        #             if is_3d:
        #                 a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
        #             else:
        #                 a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        #         pose.append(a)
        #     # ax.axis([-1, 1, -1, 1])
        #     ax.axis([0, 1000, 0, 1000])
        #     if is_3d:
        #         ax.set_zlim3d(-1, 1)
        #     for t in range(T):
        #         for m in range(M):
        #             for i, (v1, v2) in enumerate(edge):
        #                 # x1 = data[0, :2, t, v1, m]
        #                 # x2 = data[0, :2, t, v2, m]
        #                 x1 = data[0][0, :2, t, v1]
        #                 x2 = data[0][0, :2, t, v2]
        #                 # if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
        #                 if (x1[0] * x1[1] * x2[0] * x2[1] != 0) and (x1[0] > 0 and  x1[1] > 0 and  x2[0] > 0 and x2[1] > 0):
        #                     pose[m][i].set_xdata(data[0][0, 0, t, [v1, v2]])
        #                     pose[m][i].set_ydata(data[0][0, 1, t, [v1, v2]])
        #                     # fig.canvas.draw()
        #                     # plt.savefig('/home/hao/project_2021_2/MS-G3D-Eval/imgs/robot/' + str(t) + '.jpg')
        #                     # plt.pause(0.01)
        #                     if is_3d:
        #                         pose[m][i].set_3d_properties(data[0][0, 2, t, [v1, v2]])
        #             # plt.axis('equal')
        #             fig.canvas.draw()
        #             plt.savefig('/home/hao/project_2021_2/MS-G3D-Eval/imgs/robot/' + str(t) + '.jpg')
        #             fig.canvas.flush_events()
        #             plt.pause(0.01)
        # else:
        #     plot_pose(data, 4)

def draw_line(x1, y1, x2, y2):
    # Use original Image coordinate system
    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        plt.plot([x1, x2], [y1, y2])

def draw_pose(x_raw, y_raw):
    draw_line(x_raw[15], y_raw[15], x_raw[13], y_raw[13])
    draw_line(x_raw[13], y_raw[13], x_raw[11], y_raw[11])
    draw_line(x_raw[16], y_raw[16], x_raw[14], y_raw[14])
    draw_line(x_raw[14], y_raw[14], x_raw[12], y_raw[12])
    draw_line(x_raw[11], y_raw[11], x_raw[12], y_raw[12])
    draw_line(x_raw[5], y_raw[5], x_raw[11], y_raw[11])
    draw_line(x_raw[6], y_raw[6], x_raw[12], y_raw[12])
    draw_line(x_raw[5], y_raw[5], x_raw[6], y_raw[6])
    draw_line(x_raw[5], y_raw[5], x_raw[7], y_raw[7])
    draw_line(x_raw[6], y_raw[6], x_raw[8], y_raw[8])
    draw_line(x_raw[7], y_raw[7], x_raw[9], y_raw[9])
    draw_line(x_raw[8], y_raw[8], x_raw[10], y_raw[10])
    draw_line(x_raw[1], y_raw[1], x_raw[2], y_raw[2])
    draw_line(x_raw[0], y_raw[0], x_raw[1], y_raw[1])
    draw_line(x_raw[0], y_raw[0], x_raw[2], y_raw[2])
    draw_line(x_raw[1], y_raw[1], x_raw[3], y_raw[3])
    draw_line(x_raw[2], y_raw[2], x_raw[4], y_raw[4])
    draw_line(x_raw[3], y_raw[3], x_raw[5], y_raw[5])
    draw_line(x_raw[4], y_raw[4], x_raw[6], y_raw[6])


def plot_pose(data, num_clip):

    plt.clf()
    plt.cla()
    plt.close()

    fig = plt.figure()

    for i in range(data[0][num_clip].shape[1]):
    # for i, _ in enumerate(data[0][num_clip]):
        # orig from Hao
        # x_raw = data[0][num_clip][i][:, 0]
        # y_raw = data[0][num_clip][i][:, 1]

        #test
        x_raw = data[0][num_clip][0][i]
        y_raw = data[0][num_clip][1][i]
        chosen_points = (x_raw > 0) & (y_raw > 0)
        x = x_raw[chosen_points]
        y = y_raw[chosen_points]
        label = data[1]['action']

        # Method 1: add '-' to y value to convert into negative values
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(x, -y)
        # draw_pose(x_raw, -y_raw)
        # plt.title(label)
        # # preserve aspect ratio of the plot
        # plt.axis('equal')
        # plt.show(block=False)

        # Method 2: invert Y axis
        ax = plt.gca()  # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
        ax.xaxis.tick_top()  # and move the X-Axis
        ax.yaxis.tick_left()  # remove right y-Ticks
        plt.scatter(x, y)
        draw_pose(x_raw, y_raw)
        plt.title(label + "   Frame: " + str(i))

        # auto-adjust axis x-y scale
        plt.axis('equal')

        # Show in pycharm
        # plt.show(block=False)

        # Store pictures into directory
        # figname = 'fig_{}.png'.format(i)
        # dest = os.path.join('picture', figname)
        # plt.savefig(dest)  # write image to file
        plt.savefig('/home/hao/project_2021_2/MS-G3D-Eval/imgs/robot/' + str(i) + '.jpg')
        plt.clf()

if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'

    # data_path_ntu = "../data/ntu/xview/val_data_joint.npy"
    # label_path_ntu = "../data/ntu/xview/val_label.pkl"
    # data_ntu = np.load(data_path_ntu, allow_pickle=True)
    # with open(label_path_ntu, 'rb') as f_ntu:
    #     # unique_label = pickle.load(f_ntu)
    #     sample_name_ntu, label_path_ntu = pickle.load(f_ntu)

    data_path = "../data/robot/X_global_data.npy"
    label_path = "../data/robot/Y_global_data.json"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S001C001P003R001A001', graph=graph, is_3d=False)

    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)








