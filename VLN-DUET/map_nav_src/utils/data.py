import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import csv
import base64
import csv
import random
csv.field_size_limit(100000000)
TSV_FIELDNAMES = ['scanId', 'viewpointId', 'step_idx','instr_id', 'features']
TSV_FIELDNAMES_TRAINAUG = ['scanId', 'viewpointId', 'path_id', 'step_info', 'caption_idx', 'features']

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size, train_aug_ft_file=None, prevalent_aug_ft_file=None):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self.train_aug_ft_file = train_aug_ft_file
        self._feature_store_aug_train = {}
        self.vps = {}
        self._feature_store_aug_rand = {}

        if self.train_aug_ft_file != 'default' and self.train_aug_ft_file is not None:
            print('train_aug_img_ft:', self.train_aug_ft_file)
            # count = 0
            with open(self.train_aug_ft_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES_TRAINAUG)
                for row in reader:
                    # e.g. 3487_rwt_1
                    current_path_id = row['path_id'].split('pathid_')[1] + '_rwt_' + row['caption_idx'].split('caption_')[1]
                    key = '%s_%s_%s' % (row['scanId'], row['viewpointId'], current_path_id)
                    self._feature_store_aug_train[key] = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape((36, self.image_feat_size))
                    rand_ft_key = '%s_%s' % (row['scanId'], row['viewpointId'])
                    if rand_ft_key not in self.vps:
                        self.vps[rand_ft_key] = 0
                    self.vps[rand_ft_key] += 1
                    key = '%s_%s_%s' % (row['scanId'], row['viewpointId'], str(self.vps[rand_ft_key]))
                    self._feature_store_aug_rand[key] = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape((36, self.image_feat_size))
            print('ok')


    def get_image_feature(self, scan, viewpoint, sd_path_id=None):
        if sd_path_id is not None: 
            key = '%s_%s_%s' % (scan, viewpoint, sd_path_id)
            if key in self._feature_store_aug_train:
                ft = self._feature_store_aug_train[key]
            else:
                key = '%s_%s' % (scan, viewpoint)
                if key in self._feature_store:
                    ft = self._feature_store[key]
                else:
                    with h5py.File(self.img_ft_file, 'r') as f:
                        # ft = np.frombuffer(base64.b64decode(f[key][...]), dtype=np.float32).reshape((36, self.image_feat_size))
                        # ft = np.frombuffer(f[key][...], dtype=np.float32).reshape((36, self.image_feat_size))
                        ft = np.frombuffer(f[key][...], dtype=np.float32).reshape((36, self.image_feat_size))
                        # ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                        self._feature_store[key] = ft     
            rand_ft_key = '%s_%s' % (scan, viewpoint)
            if rand_ft_key in self.vps:
                ft_num = self.vps[rand_ft_key]
                rand_num = random.randrange(ft_num) + 1
                key = '%s_%s_%s' % (scan, viewpoint, str(rand_num))
                ft = self._feature_store_aug_rand[key]
            else:
                key = '%s_%s' % (scan, viewpoint)
                if key in self._feature_store:
                    ft = self._feature_store[key]
                else:
                    with h5py.File(self.img_ft_file, 'r') as f:
                        ft = np.frombuffer(f[key][...], dtype=np.float32).reshape((36, self.image_feat_size))
                        self._feature_store[key] = ft
        else:
            key = '%s_%s' % (scan, viewpoint)
            if self.img_ft_file.endswith('.tsv'):
                if key in self._feature_store:
                    ft = self._feature_store[key]
                else:
                    with open(self.img_ft_file, "r", encoding="utf-8") as tsv_in_file:
                        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
                        for row in reader:
                            if row['scanId'] == scan and row['viewpointId'] == viewpoint:
                                ft = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape((36, self.image_feat_size)) # (VIEWPOINT_SIZE, FEATURE_SIZE)
                                self._feature_store[key] = ft
            else:
                if key in self._feature_store:
                    ft = self._feature_store[key]
                else:
                    with h5py.File(self.img_ft_file, 'r') as f:
                        ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                        self._feature_store[key] = ft
        return ft

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

