#!/usr/bin/env python3
import cv_bridge
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from line_segment_detector.msg import LineSeg, LineSegArray
import numpy as np
import cv2

def convert_multiarray_to_numpy_2d(msg_linesegments):
    return np.reshape(msg_linesegments.data, (-1, msg_linesegments.layout.dim[0].size))

class ImgBlend(object):
    def __init__(self):
        self.id = -1
        self.pub = rospy.Publisher('semantic_blend', Image, queue_size = 3)
        print("....publisher initialized")

        self.img_raw = Image()
        self.img_semantic = Image()
        self.img_blend = Image()
        rospy.Subscriber('/rear_cam/image_raw', Image, self.callback_img_raw)
        rospy.Subscriber('semantic_color', Image, self.callback_img_semantic)
        rospy.Subscriber('line_segments', LineSegArray, self.callback_line_seg)
        rospy.spin()
    def add_weighted(self, img1, weight1, img2, weight2):
        if img1 is None or img2 is None:
            return
        if np.shape(img1) == np.shape(img2) and weight1+weight2 == 1.0:
            img_new = np.add(img1*weight1, img2*weight2)
            return np.uint8(img_new)
        else:
            min_height = np.minimum(np.shape(img1)[0], np.shape(img2)[0])
            max_height = np.maximum(np.shape(img1)[0], np.shape(img2)[0])
            max_width = np.maximum(np.shape(img1)[1], np.shape(img2)[1])
            min_width = np.minimum(np.shape(img1)[1], np.shape(img2)[1])

            padding = int( (max_width - min_width)/2 )
            img2 = np.pad(img2, ((0,0), (63, 64)), 'constant', constant_values = 0)
            print("IMG SHAPE")
            print(np.shape(img2))

            row_delete_top = int( (max_height - min_height)/2 )
            img2 = img2[row_delete_top:480,:] 
            img_new = np.add(img1*weight1, img2*weight2)
            return np.uint8(img_new)
    def callback_img_raw(self, msg):
        print("!!!! img_raw received.")
        self.img_raw = cv_bridge.imgmsg_to_cv2(msg)
        if self.img_semantic is None or self.img_raw is None:    
            return    
        self.img_blend = self.add_weighted(self.img_raw, 0.8, self.img_semantic, 0.2)
        
        print((self.img_blend is None))
        if self.img_blend is None:
            return
        #print("####img blended")
        #self.pub.publish(cv_bridge.cv2_to_imgmsg(self.img_blend))
    def callback_img_semantic(self, msg):
        print("!!!! img_semantic received.")
        self.img_semantic = cv_bridge.imgmsg_to_cv2(msg)
        if self.img_semantic is None or self.img_raw is None:    
            return
        self.img_blend = self.add_weighted(self.img_raw, 0.8, self.img_semantic, 0.2)
        if self.img_blend is None:
            return
        #print("####img blended")
        #self.pub.publish(cv_bridge.cv2_to_imgmsg(self.img_blend))
    def callback_line_seg(self, msg):
        print("!!!! line segments received.")
        if self.img_blend is None or self.img_raw is None:
            return
        img_blank = 128*np.ones(self.img_raw.shape)
        print(img_blank.shape)
        for i_seg in range(len(msg.line_segments)):
            #print(line_segments[i_seg])
            img_blank = cv2.line(img_blank,
                            (int(msg.line_segments[i_seg].start.y), int(msg.line_segments[i_seg].start.x)),
                            (int(msg.line_segments[i_seg].end.y), int(msg.line_segments[i_seg].end.x)),
                            (0, 255, 0), 3)
        self.img_blend = self.add_weighted(self.img_blend, 0.5, img_blank, 0.5)
        print("####img LINESEG blended")
        self.pub.publish(cv_bridge.cv2_to_imgmsg(self.img_blend))

if __name__=='__main__':
    # ...setup stuff...
    print("....init node")
    rospy.init_node('blend_img')
    try:
        viz_pub = ImgBlend()
    except rospy.ROSInterruptException: pass 
