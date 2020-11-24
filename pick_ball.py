import sys
# sys.path.append('/home/ainz/Desktop/Demo/dji_python_sdk/build/lib.linux-aarch64-3.6')
from robomaster import robot
import robomaster
import numpy as np
import signal
import cv2
import time
import imutils
import PID
from tf_my import MyTf
import math
import threading

pos = [0, 0]
sub_yaw = 0
sub_position_x = 0
sub_position_y = 0


def move_with_wheel_speed(x=0.0, y=0.0, yaw=0.0):
    if x > 0.2:
        x = 0.2
    elif x < -0.2:
        x = -0.2
    if y > 0.2:
        y = 0.2
    elif y < -0.2:
        y = -0.2
    if yaw > 50:
        yaw = 50
    elif yaw < -50:
        yaw = -50
    # print(y)
    chassis.drive_speed(x, y, yaw)

#


def __tag_scan_task():
    global find_ball_flag
    global pos
    global robot_is_alive

    def nothing(x):
        pass
        # Create trackbar

    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('lh', 'Trackbar', 30, 255, nothing)
    cv2.createTrackbar('ls', 'Trackbar', 69, 255, nothing)
    cv2.createTrackbar('lv', 'Trackbar', 116, 255, nothing)
    cv2.createTrackbar('hh', 'Trackbar', 68, 255, nothing)
    cv2.createTrackbar('hs', 'Trackbar', 255, 255, nothing)
    cv2.createTrackbar('hv', 'Trackbar', 255, 255, nothing)
    while robot_is_alive:
        frame = cam.read_cv2_image(strategy='newest')
        frame = cv2.resize(frame, (640, 360))
        # 720*1280
        if frame.any():
            last_time = time.time()
            find_ball_flag, pos, result_img = process(frame)
            # print(time.time()-last_time)
            cv2.imshow("result", result_img)

            cv2.waitKey(1)
    print("exit threading")
    cv2.destroyAllWindows()


def process(image):
    blurred = cv2.medianBlur(image, 5)
    # RGB to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([46, 218, 116])
    high_hsv = np.array([85, 255, 255])

    # Get trackbar position
    low_hsv[0] = cv2.getTrackbarPos('lh', 'Trackbar')
    low_hsv[1] = cv2.getTrackbarPos('ls', 'Trackbar')
    low_hsv[2] = cv2.getTrackbarPos('lv', 'Trackbar')
    high_hsv[0] = cv2.getTrackbarPos('hh', 'Trackbar')
    high_hsv[1] = cv2.getTrackbarPos('hs', 'Trackbar')
    high_hsv[2] = cv2.getTrackbarPos('hv', 'Trackbar')

    mask = cv2.inRange(hsv, low_hsv, high_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask', mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    pos = [0, 0]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        x = int(x)
        y = int(y)
        radius = int(radius)
        center = (x, y)
        if 10 < radius < 65:
            cv2.circle(image, center, radius, (0, 255, 0), 4)
            cv2.circle(image, center, 5, (0, 128, 255), -1)
            cv2.putText(image, '{} {} {}'.format(x, y, radius),
                        center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            pos[0] = x
            pos[1] = y
            ret = True

        else:
            # print('No ball recognized')
            ret = False
    else:
        # print('No ball recognized')
        ret = False
    return ret, pos, image


first_init_pos = True


def sub_position_info_handler(position_info):
    global first_init_pos
    global sub_position_x, sub_position_y
    global old2origin_frame, old2now_frame, origin2now_frame

    raw_position_x = position_info[0]
    raw_position_y = -position_info[1]
    if first_init_pos:
        old2origin_frame.origin = np.array([raw_position_x, raw_position_y, 0])
        first_init_pos = False
    else:

        old2now_frame.origin = np.array([raw_position_x, raw_position_y, 0])
        origin2now_frame.homo = np.matmul(np.linalg.inv(old2origin_frame.homo),
                                          old2now_frame.homo)

        sub_position_x = origin2now_frame.get_origin()[0]
        sub_position_y = origin2now_frame.get_origin()[1]


first_init_yaw = True


def sub_attitude_info_handler(attitude_info):
    global sub_yaw
    global first_init_yaw
    global old2origin_frame, old2now_frame, origin2now_frame
    raw_yaw = -attitude_info[0]
    if first_init_pos:
        old2origin_frame.RPY = np.array([0, 0, raw_yaw])
        first_init_yaw = False
    else:
        old2now_frame.RPY = np.array([0, 0, raw_yaw])

        origin2now_frame.homo = np.matmul(np.linalg.inv(old2origin_frame.homo),
                                          old2now_frame.homo)
        sub_yaw = origin2now_frame.get_RPY()[2]
        # print('sub yaw %f'%sub_yaw)


def out_limit(value, maxout, minout):
    if (value > maxout):
        value = maxout
    elif (value < minout):
        value = minout
    return value


def fsm_task():
    global fsm_status_num
    if fsm_status_num == 0:
        fsm_reset_arm()
    elif fsm_status_num == 1:
        fsm_find_ball()
    elif fsm_status_num == 3:
        fsm_go_home()
    elif fsm_status_num == 4:
        fsm_rotate_and_put_down()
    elif fsm_status_num == 2:
        fsm_grip_ball()


def fsm_reset_arm():
    global fsm_status_num
    global sub_yaw
    yaw = sub_yaw
    chassis.move(z=-yaw).wait_for_completed()
    arm.moveto(x=90, y=60).wait_for_completed()
    arm.moveto(x=180, y=80).wait_for_completed()
    arm.moveto(x=190, y=-60).wait_for_completed()

    gripper.open()
    time.sleep(2)
    fsm_status_num += 1


calib_complete_success_times = 0


def fsm_find_ball():
    global fsm_status_num
    global pos
    global find_ball_flag
    global calib_complete_success_times

    if find_ball_flag is True:
        x_pos = pos[0]
        y_pos = pos[1]

        # if abs(325-x_pos) >100:#角度过大，先旋转
        #     if (245 - y_pos) >-20:
        #     pid_angle.update(325, x_pos)
        #     xout = pid_x.output
        #     # move_with_wheel_speed(0, 0, -yout)
        goal_x = 320
        goal_y = 260
        pid_angle.update(goal_x, x_pos)
        pid_x.update(goal_y, y_pos)
        xout = pid_x.output
        yout = pid_angle.output
        xout = out_limit(xout, 0.4, -0.4)
        yout = out_limit(yout, 50, -50)

        # print('x_porobomasters:%d y_pos:%d' % (x_pos, y_pos))
        if abs(goal_x - x_pos) < 11 and abs(goal_y - y_pos) < 10:
            calib_complete_success_times += 1
            move_with_wheel_speed(0, 0, -0)
        else:
            calib_complete_success_times = 0
        if calib_complete_success_times > 20:
            fsm_status_num += 1
            move_with_wheel_speed(0, 0, -0)
            gripper.open()
            time.sleep(2)
            # arm.moveto(x=180, y=80).wait_for_completed()
            # arm.moveto(x=90, y=60).wait_for_completed()
        move_with_wheel_speed(xout, 0, -yout)
    else:
        move_with_wheel_speed(0, 0, -0)


def fsm_grip_ball():
    global fsm_status_num
    
    # arm.moveto(x=200, y=-10).wait_for_completed()
    gripper.open()
    time.sleep(2)
    gripper.close()
    time.sleep(2)
    arm.moveto(x=90, y=60).wait_for_completed()
    fsm_status_num += 1


def fsm_go_home():
    global fsm_status_num
    global sub_yaw, sub_position_x, sub_position_y
    yaw = sub_yaw
    dxy_vector = np.array([sub_position_x, -sub_position_y])
    trans_matrix = np.array([
        [math.cos(yaw / 57.3), -math.sin(yaw / 57.3)],
        [math.sin(yaw / 57.3), math.cos(yaw / 57.3)]
    ])

    dxy_vector = np.matmul(trans_matrix, dxy_vector)
    # print(dxy_vector)
    if abs(yaw) > 20:
        move_with_wheel_speed(0, 0, yaw * 2)
    elif abs(dxy_vector[1]) > 0.03 or abs(yaw) > 5 or abs(dxy_vector[0]) > 0.03:
        move_with_wheel_speed(-dxy_vector[0]
                              * 2.0, -3.0 * dxy_vector[1], yaw * 3)
    else:
        move_with_wheel_speed(0, 0, 0)
        fsm_status_num += 1


def fsm_rotate_and_put_down():
    global fsm_status_num
    global sub_yaw
    yaw = sub_yaw
    move_with_wheel_speed(0, 0, 80)
    # print(yaw)
    if -180 < yaw < -170 or 180 > yaw > 170:
        move_with_wheel_speed(0, 0, 0)
        arm.moveto(x=180, y=80).wait_for_completed()
        arm.moveto(x=190, y=-60).wait_for_completed()
        gripper.open()
        time.sleep(2)
        arm.moveto(x=180, y=80).wait_for_completed()
        fsm_status_num = 0



def hit_callback(sub_info):
    # 被击打装甲的ID，被击打类型
    armor_id, hit_type = sub_info
    print("hit event: hit_comp:{0}, hit_type:{1}".format(armor_id, hit_type))

    # gripper.open()
if __name__ == '__main__':
    fsm_status_num = 0
    old2origin_frame = MyTf()
    origin2now_frame = MyTf()
    old2now_frame = MyTf()
    # EP_init
    # change this ip address to yours
    # robomaster.config.ROBOT_IP_STR = "192.168.0.108"
    ep_robot = robomaster.robot.Robot()
    # ep_robot.initialize(conn_type="rndis")
    ep_robot.initialize(conn_type='sta')
    # 模块初始化
    cam = ep_robot.camera
    arm = ep_robot.robotic_arm
    gripper = ep_robot.gripper
    chassis = ep_robot.chassis
    vision = ep_robot.vision
    led = ep_robot.led
    ep_armor = ep_robot.armor

    cam.start_video_stream(False)
    # Initialize PID parameters
    pid_angle = PID.PID(1, 0, 0, 50, 0)
    pid_x = PID.PID(0.002, 0.0, 0, 0.3, 0.0)

    find_ball_flag = False

    chassis.sub_attitude(50, sub_attitude_info_handler)
    chassis.sub_position(0, 50, sub_position_info_handler)
    # 设置所有装甲灵敏度为 5
    ep_armor.set_hit_sensitivity(comp="all", sensitivity=5)

    # 订阅装甲被击打的事件
    # ep_armor.sub_hit_event(hit_callback)
    robot_is_alive = True

    tag_socket_recv_thread = threading.Thread(target=__tag_scan_task)
    tag_socket_recv_thread.start()
    time.sleep(1)

    def exit(signum, frame):
        global robot_is_alive
        print("Closing")

        robot_is_alive = False
        print(robot_is_alive)
        tag_socket_recv_thread.join()
        cam.stop_video_stream()
        ep_robot.close()

    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)

    gripper_over = False

    while robot_is_alive:
        fsm_task()
        time.sleep(0.03)