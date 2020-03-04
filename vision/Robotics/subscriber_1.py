import rospy
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge, CvBridgeError

path = Path()

def image_sub(data):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
 
    (rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
    #    cv2.circle(cv_image, (50,50), 10, 255)
    #print(cv_image)
    #cv2.imwrite("Image.jpeg", cv_image)
    
def odometryCb(data):
    global path
    path.header = data.header
    pose = PoseStamped()
    pose.header = data.header
    pose.pose = data.pose.pose
    path.poses.append(pose)
    print path.poses
    
def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/mybot/camera1/image_raw", Image, image_sub)
    rospy.Subscriber('odom',Odometry,odometryCb)
    rospy.spin()

if __name__ == '__main__':
    listener()
