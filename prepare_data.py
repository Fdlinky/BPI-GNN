import pandas as pd
import numpy as np
import os

class Sample:
    def __init__(self, sample_id, func_conn_path=None, stru_conn_path=None):
        self.sample_id = sample_id
        self.func_conn_path = func_conn_path
        self.stru_conn_path = stru_conn_path
        self.func_conn = None
        self.stru_conn = None
        self.cortical_thickness = pd.DataFrame()
        self.surface_area = pd.DataFrame()
        self.volume_size = pd.DataFrame()

    def load_func_conn(self):
        if self.func_conn_path and os.path.exists(self.func_conn_path):
            self.func_conn = np.loadtxt(self.func_conn_path)
        else:
            self.func_conn = None
        return self.func_conn

    def load_stru_conn(self):
        if self.stru_conn_path and os.path.exists(self.stru_conn_path):
            self.stru_conn = np.loadtxt(self.stru_conn_path)
        else:
            self.stru_conn = None
        return self.stru_conn

    def get_func_conn(self):
        if self.func_conn is None:
            self.load_func_conn()
        return self.func_conn

    def get_stru_conn(self):
        if self.stru_conn is None:
            self.load_stru_conn()
        return self.stru_conn

    def set_cortical_thickness(self, thickness_df):
        self.cortical_thickness = thickness_df

    def set_surface_area(self, area_df):
        self.surface_area = area_df

    def set_volume_size(self, volume_df):
        self.volume_size = volume_df

    def get_node_features(self):
        return {
            'cortical_thickness': self.cortical_thickness,
            'surface_area': self.surface_area,
            'volume_size': self.volume_size
        }