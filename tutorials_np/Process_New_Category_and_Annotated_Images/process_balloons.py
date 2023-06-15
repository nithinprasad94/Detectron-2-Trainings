# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Register new dataset with Detectron 2
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "balloon_json/balloons_1.json", "/home/robot/Documents/detectron_2_direct/detectron2/tutorials_np/Process_New_Category_and_Annotated_Images/balloon_dataset/balloon/train")
register_coco_instances("my_dataset_val", {}, "balloon_json/balloons_validation_2.json", "/home/robot/Documents/detectron_2_direct/detectron2/tutorials_np/Process_New_Category_and_Annotated_Images/balloon_dataset/balloon/validate")

### Verify Dataset is in correct format
##dataset_dicts = get_balloon_dicts("balloon/train")
##for d in random.sample(dataset_dicts, 3):
##    img = cv2.imread(d["file_name"])
##    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
##    out = visualizer.draw_dataset_dict(d)
##    cv2_imshow(out.get_image()[:, :, ::-1])

## TRAIN?
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

##import torch
##foo = torch.tensor([1,2,3])
##foo = foo.to('cuda')
##torch.cuda.empty_cache()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
