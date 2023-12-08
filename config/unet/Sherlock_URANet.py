_base_ = [
    '../_base_/models/URANet.py', 
    '../_base_/datasets/WaterDataset_360x640.py',
    #'../_base_/Runtime_trainWater.py',
    #'../_base_/schedules/Schedule_300ep_uranet.py',
]

model = dict()
