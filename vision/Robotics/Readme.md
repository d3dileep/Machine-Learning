This file helps to subscribe ROS camera data and Odometry data. Camera data can be viewed, saved, or processed further.
Change 'Odom' and '/mybot/camera1/image_raw' to your respective rostopic for odometry and camera.

    rospy.Subscriber("/mybot/camera1/image_raw", Image, image_sub)
    rospy.Subscriber('odom',Odometry,odometryCb)
