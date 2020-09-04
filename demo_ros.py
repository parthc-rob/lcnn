#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    demo.py [options] <yaml-config> <checkpoint> <images>...
    demo.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <images>                      Path to images

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
"""
import rospy
from sensor_msgs.msg import Image
import cv_bridge
from line_segment_detector.msg import LineSeg, LineSegArray
from geometry_msgs.msg import Point

import os
import os.path as osp
import pprint
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)

class LineSegmentDetector(object):
    def __init__(self):

        self.pub = rospy.Publisher('line_segments', LineSegArray, queue_size = 3)
        rospy.Subscriber('rear_cam/image_raw', Image, self.callback)
        self.img_input = Image()

        args = docopt(__doc__)
        config_file = args["<yaml-config>"] or "config/wireframe.yaml"
        C.update(C.from_yaml(filename=config_file))
        M.update(C.model)
        pprint.pprint(C, indent=4)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        device_name = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(0)
            print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        else:
            print("CUDA is not available")
        self.device = torch.device(device_name)
        checkpoint = torch.load(args["<checkpoint>"], map_location=self.device)

        #self.model = C.model
        # Load model
        self.model = lcnn.models.hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
        self.model = MultitaskLearner(self.model)
        self.model = LineVectorizer(self.model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        rospy.spin()
    #for imname in args["<images>"]:
        #print(f"Processing {imname}")
    def callback(self, msg):
        self.img_input = cv_bridge.imgmsg_to_cv2(msg)
        if self.img_input is None:
            return
        #im = skimage.io.imread(self.img_input)
        im = self.img_input
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(self.device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(self.device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
                },
                "mode": "testing",
            }
            H = self.model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        segment_array = LineSegArray()
        segment_array_list = list()
        for idx_seg in range(nlines.shape[0]):
            segment = LineSeg()
            segment.start = Point( nlines[idx_seg, 0, 0], nlines[idx_seg, 0, 1], 0.)
            segment.end = Point( nlines[idx_seg, 1, 0], nlines[idx_seg, 1, 1], 0.)
            segment_array_list.append(segment)
        segment_array.line_segments = segment_array_list
        self.pub.publish(segment_array)

if __name__ == "__main__":
    print("... init node")
    rospy.init_node('line_segment_detector')
    try:
        lineseg_pub = LineSegmentDetector()
    except rospy.ROSInterruptException: pass
