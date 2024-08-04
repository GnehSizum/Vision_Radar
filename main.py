import threading
import time
import math
from collections import deque
import serial
from information_ui import draw_information_ui
from gx_camera import *
import gxipy as gx
from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control
from MvImport.MvCameraControl_class import *
import cv2
import numpy as np
from detect_function import YOLOv5Detector
from RM_serial_py.ser_api import build_data_radar, build_send_packet, receive_packet, Radar_decision, \
    build_data_decision


state = 'R'  # R:红方/B:蓝方

if state == 'R':
    loaded_arrays = np.load('arrays_test_red.npy')  # 加载标定好的仿射变换矩阵
    map_image = cv2.imread("images/map_red.jpg")  # 加载红方视角地图
    mask_image = cv2.imread("images/map_mask.jpg")  # 加载红发落点判断掩码
    hide_mask = cv2.imread('images/hide_mask.jpg')
else:
    loaded_arrays = np.load('arrays_test_blue.npy')  # 加载标定好的仿射变换矩阵
    map_image = cv2.imread("images/map_blue.jpg")  # 加载蓝方视角地图
    mask_image = cv2.imread("images/map_mask.jpg")  # 加载蓝方落点判断掩码
    hide_mask = cv2.imread('images/hide_mask.jpg')

# 导入战场每个高度的不同仿射变化矩阵
M_height_r = loaded_arrays[1]  # R型高地
M_height_g = loaded_arrays[2]  # 环形高地
M_ground = loaded_arrays[0]  # 地面层、公路层

# 确定地图画面像素，保证不会溢出
height, width = mask_image.shape[:2]
height -= 1
width -= 1

# 初始化战场信息UI（标记进度、双倍易伤次数、双倍易伤触发状态）
information_ui = np.zeros((500, 420, 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标（用于触发双倍易伤）
chances_flag = 1  # 双倍易伤触发标志位，需要从1递增，每小局比赛会重置，所以每局比赛要重启程序
progress_list = [-1, -1, -1, -1, -1, -1]  # 标记进度列表

# 加载战场地图
map_backup = cv2.imread("images/map.jpg")
map = map_backup.copy()

# 盲区预测次数
guess_value = {
    "B1": 0,
    "B2": 0,
    "B3": 0,
    "B4": 0,
    "B5": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R3": 0,
    "R4": 0,
    "R5": 0,
    "R7": 0
}

# 当前标记进度（用于判断是否预测正确正确）
mark_progress = {
    "B1": 0,
    "B2": 0,
    "B3": 0,
    "B4": 0,
    "B5": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R3": 0,
    "R4": 0,
    "R5": 0,
    "R7": 0
}

# 机器人名字对应ID
mapping_table = {
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4": 4,
    "R5": 5,
    "R6": 6,
    "R7": 7,
    "B1": 101,
    "B2": 102,
    "B3": 103,
    "B4": 104,
    "B5": 105,
    "B6": 106,
    "B7": 107
}


# 预测点索引
guess_index = {
    'B1': 0,
    'B2': 0,
    'B3': 0,
    'B4': 0,
    'B5': 0,
    'B7': 0,
    'R1': 0,
    'R2': 0,
    'R3': 0,
    'R4': 0,
    'R5': 0,
    'R7': 0,
}

guess_table_B = {
    "G0": [(560, 650), (560, 850)],
    "G1": [(160, 370), (160, 170)],
    "G2": [(800, 650), (800, 850)],
    "G3": [(870, 1230), (735, 1035)]
}

guess_table_R = {
    "G0": [(2240, 850), (2240, 650)],
    "G1": [(2640, 1130), (2640, 1130)],
    "G2": [(2000, 850), (2000, 650)],
    "G3": [(1930, 270), (2065, 4650)]
}


# 机器人坐标滤波器（滑动窗口均值滤波）
class Filter:
    def __init__(self, window_size, max_inactive_time=2.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.data = {}  # 存储不同机器人的数据
        self.window = {}  # 存储滑动窗口内的数据
        self.last_update = {}  # 存储每个机器人的最后更新时间

    # 添加机器人坐标数据
    def add_data(self, name, x, y, conf, threshold=100000.0):  # 阈值单位为mm，实测没啥用，不如直接给大点
        print('name: ', name)
        name0 = name
        flag = 0
        if name.startswith('R'):
            new_name = name.replace('R', 'B') 
        if name.startswith('B'):
            new_name = name.replace('B', 'R')
        
        if name not in self.data:
            # 如果实体名称不在数据字典中，初始化相应的deque。
            self.data[name] = deque(maxlen=self.window_size)
            self.window[name] = deque(maxlen=self.window_size)
        if conf > 0.82:
            if len(self.window[name]) >= 2:
                # 计算当前坐标与前一个坐标的均方
                msd = sum((a - b) ** 2 for a, b in zip((x, y), self.window[name][-1])) / 2.0
                if msd > threshold:
                    # 如果均方差超过阈值，可能是异常值，不将其添加到数据中
                    return flag
        else:
            if new_name not in self.data:
                # 如果实体名称不在数据字典中，初始化相应的deque。
                self.data[new_name] = deque(maxlen=self.window_size)
                self.window[new_name] = deque(maxlen=self.window_size)
            if len(self.window[new_name]) >= 2:
                dis0 = math.sqrt(sum((a - b) ** 2 for a, b in zip((x, y), self.window[new_name][-1])))
                # print('dis0: ', dis0)
                if dis0 < 80:
                    self.data[name].clear()
                    self.window[name].clear()
                    name0 = new_name
                    flag = 1

        # 将坐标数据添加到数据字典和滑动窗口中。
        self.data[name0].append((x, y))
        self.window[name0].append((x, y))
        self.last_update[name0] = time.time()  # 更新最后更新时间
        return flag

    # 过滤计算滑动窗口平均值
    def filter_data(self, name):
        if name not in self.data:
            return None

        if len(self.window[name]) < self.window_size:
            return None  # 不足以进行滤波

        # 计算滑动窗口内的坐标平均值
        x_avg = sum(coord[0] for coord in self.window[name]) / self.window_size
        y_avg = sum(coord[1] for coord in self.window[name]) / self.window_size

        return x_avg, y_avg

    # 获取所有机器人坐标
    def get_all_data(self):
        filtered_d = {}
        for name in self.data:
            if name in self.last_update:
                # 超过max_inactive_time没识别到机器人将会清空缓冲区
                if time.time() - self.last_update[name] > self.max_inactive_time:
                    self.data[name].clear()
                    self.window[name].clear()
                else:
                    filtered_d[name] = self.filter_data(name)
        # 返回所有当前识别到的机器人及其坐标的均值
        return filtered_d


# Galaxy Camera
def gx_camera_get():
    global camera_image
    device_manager = gx.DeviceManager()
    cam = None
    dev_info_list = create_device(device_manager)
    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
    open_device(cam)
    start_acquisition(cam)

    while True:
        numpy_image = get_image(cam)
        camera_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


# 海康相机图像获取线程
def hik_camera_get():
    # 获得设备信息
    global camera_image
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    while 1:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            # sys.exit()
        else:
            print("Find %d devices!" % deviceList.nDeviceNum)
            break

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 手动选择设备
    # nConnectionNum = input("please input the number of the device to connect:")
    # 自动选择设备
    nConnectionNum = '0'
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    print(get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))

    # 设置设备的一些参数
    set_Value(cam, param_type="float_value", node_name="ExposureTime", node_value=16000)  # 曝光时间
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=17.9)  # 增益值
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    # 主动取流方式抓取图像
    stParam = MVCC_INTVALUE_EX()

    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            # 处理海康相机的图像格式为OPENCV处理的格式
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print("no data[0x%x]" % ret)


def video_capture_get():
    global camera_image
    cam = cv2.VideoCapture(0)
    while True:
        ret,img = cam.read()
        if ret:
            # camera_image = img

             # 增加图像的亮度
            bright_img = cv2.convertScaleAbs(img, alpha=1, beta=35)  # 增加亮度50
            camera_image = bright_img

            time.sleep(0.016) # 60fps


def video_test_get():
    global camera_image
    video = cv2.VideoCapture('/home/mumu/Videos/test2.avi')
    while video.isOpened():
        ret, frame = video.read()
        if not ret: 
            break
        camera_image = frame
        time.sleep(0.016)
    video.release()


# 串口发送线程
def ser_send():
    seq = 0
    global chances_flag
    global guess_value

    def return_xy_B(send_name):
        # 转换为地图坐标系
        filtered_xyz = (2800 - all_filter_data[send_name][1], all_filter_data[send_name][0])
        # 转换为裁判系统单位 cm
        ser_x = int(filtered_xyz[0])
        ser_y = int(1500 - filtered_xyz[1])
        return [ser_x, ser_y]
    
    def return_xy_R(send_name):
        # 转换为地图坐标系
        filtered_xyz = (all_filter_data[send_name][1], 1500 - all_filter_data[send_name][0])
        # 转换为裁判系统单位M
        ser_x = int(filtered_xyz[0])
        ser_y = int(1500 - filtered_xyz[1])
        return [ser_x, ser_y]

    # 发送机器人坐标
    def send_point(target_position, seq_s):
        front_time = time.time()
        # 打包坐标数据包
        data = build_data_radar(target_position)
        packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        ser1.write(packet)
        back_time = time.time()
        # 计算发送延时，动态调整
        waste_time = back_time - front_time
        # print("发送：", seq_s)
        time.sleep(0.2 - waste_time)
        return seq_s
    
    def get_guess_point(send_name):
        guess_point = [0,0]
        if state == 'B':
            if send_name == "R7":
                if guess_value[send_name] == 0:
                    guess_point = guess_table_B['G0'][0]
                    guess_value[send_name] = 1
                else:
                    guess_point = guess_table_B['G0'][1]
                    guess_value[send_name] = 0
            else:
                if guess_index[send_name] == 1:
                    if send_name == "R2":
                        guess_point = guess_table_B['G1'][0]
                    else:
                        guess_point = guess_table_B['G1'][1]
                if guess_index[send_name] == 2:
                    value = guess_value[send_name] % 2
                    guess_point = guess_table_B['G2'][value]
                    guess_value[send_name] = guess_value[send_name]+1
                    if guess_value[send_name] == 5:
                        guess_value[send_name] = -1
                if guess_index[send_name] == 3:
                    value = guess_value[send_name] % 2
                    guess_point = guess_table_B['G3'][value]
                    guess_value[send_name] = guess_value[send_name]+1
                    if guess_value[send_name] == 5:
                        guess_value[send_name] = -1
        if state == 'R':
            if send_name == "B7":
                if guess_value[send_name] == 0:
                    guess_point = guess_table_B['G0'][0]
                    guess_value[send_name] = 1
                else:
                    guess_point = guess_table_B['G0'][1]
                    guess_value[send_name] = 0
            else:
                if guess_index[send_name] == 1:
                    if send_name == "B2":
                        guess_point = guess_table_R['G1'][0]
                    else:
                        guess_point = guess_table_R['G1'][1]
                if guess_index[send_name] == 2:
                    value = guess_value[send_name] % 2
                    guess_point = guess_table_B['G2'][value]
                    guess_value[send_name] = guess_value[send_name]+1
                    if guess_value[send_name] == 5:
                        guess_value[send_name] = -1
                if guess_index[send_name] == 3:
                    value = guess_value[send_name] % 2
                    guess_point = guess_table_B['G3'][value]
                    guess_value[send_name] = guess_value[send_name]+1
                    if guess_value[send_name] == 5:
                        guess_value[send_name] = -1
        return guess_point


    time_s = time.time()
    target_last = 0  # 上一帧的飞镖目标
    while True:
        try:
            all_filter_data = filter.get_all_data()
            target_position = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
            if state == 'R':
                # 英雄
                if all_filter_data.get('B1', False):
                    target_position[0] = return_xy_B('B1')
                    # guess_index['B1'] = 0
                    guess_value['B1'] = 0
                else:
                    if guess_value['B1'] > -1:
                        target_position[0] = get_guess_point('B1')
                # 工程
                if all_filter_data.get('B2', False):
                    target_position[1] = return_xy_B('B2')
                    # guess_index['B2'] = 0
                    guess_value['B2'] = 0
                else:
                    if guess_value['B2'] > -1:
                        target_position[1] = get_guess_point('B2')
                # 步兵3号
                if all_filter_data.get('B3', False):
                    target_position[2] = return_xy_B('B3')
                    # guess_index['B3'] = 0
                    guess_value['B3'] = 0
                else:
                    if guess_value['B3'] > -1:
                        target_position[2] = get_guess_point('B3')
                # 步兵4号
                if all_filter_data.get('B4', False):
                    target_position[3] = return_xy_B('B4')
                    # guess_index['B4'] = 0
                    guess_value['B4'] = 0
                else:
                    if guess_value['B4'] > -1:
                        target_position[3] = get_guess_point('B4')
                # 步兵5号
                if all_filter_data.get('B5', False):
                    target_position[4] = return_xy_B('B5')
                    # guess_index['B5'] = 0
                    guess_value['B5'] = 0
                else:
                    if guess_value['B5'] > -1:
                        target_position[4] = get_guess_point('B5')
                # 哨兵
                if all_filter_data.get('B7', False):
                    target_position[5] = return_xy_B('B7')
                    # guess_index['B7'] = 0
                    guess_value['B7'] = 0
                else:
                    target_position[5] = get_guess_point('B7')
                seq = send_point(target_position, seq)

            if state == 'B':
                # 英雄
                if all_filter_data.get('R1', False):
                    target_position[0] = return_xy_R('R1')
                    # guess_index['R1'] = 0
                    guess_value['R1'] = 0
                else:
                    if guess_value['R1'] > -1:
                        target_position[0] = get_guess_point('R1')
                # 工程
                if all_filter_data.get('R2', False):
                    target_position[1] = return_xy_R('R2')
                    # guess_index['R2'] = 0
                    guess_value['R2'] = 0
                else:
                    if guess_value['R2'] > -1:
                        target_position[1] = get_guess_point('R2')
                # 步兵3号
                if all_filter_data.get('R3', False):
                    target_position[2] = return_xy_R('R3')
                    # guess_index['R3'] = 0
                    guess_value['R3'] = 0
                else:
                    if guess_value['R3'] > -1:
                        target_position[2] = get_guess_point('R3')
                # 步兵4号
                if all_filter_data.get('R4', False):
                    target_position[3] = return_xy_R('R4')
                    # guess_index['R4'] = 0
                    guess_value['R4'] = 0
                else:
                    if guess_value['R4'] > -1:
                        target_position[3] = get_guess_point('R4')
                # 步兵5号
                if all_filter_data.get('R5', False):
                    target_position[4] = return_xy_R('R5')
                    # guess_index['R5'] = 0
                    guess_value['R5'] = 0
                else:
                    if guess_value['R5'] > -1:
                        target_position[4] = get_guess_point('R5')
                # 哨兵
                if all_filter_data.get('R7', False):
                    target_position[5] = return_xy_R('R7')
                    # guess_index['R7'] = 0
                    guess_value['R7'] = 0
                else:
                    target_position[5] = get_guess_point('R7')
                seq = send_point(target_position, seq)
            
            # 判断飞镖的目标是否切换，切换则尝试发动双倍易伤
            if target != target_last and target != 0:
                target_last = target
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 发送时间间隔为10秒
                    if time_e - time_s > 10:
                        print("请求双倍触发")
                        data = build_data_decision(chances_flag, state)
                        packet, seq = build_send_packet(data, seq, [0x03, 0x01])
                        # print(packet.hex(),chances_flag,state)
                        ser1.write(packet)
                        print("请求成功", chances_flag)
                        # 更新标志位
                        chances_flag += 1
                        if chances_flag >= 3:
                            chances_flag = 1

                        time_s = time.time()
        except Exception as r:
            print('未知错误 %s' % (r))


# 裁判系统串口接收线程
def ser_receive():
    global progress_list  # 标记进度列表
    global double_vulnerability_chance  # 拥有双倍易伤次数
    global opponent_double_vulnerability  # 双倍易伤触发状态
    global target  # 飞镖当前目标
    progress_cmd_id = [0x02, 0x0C]  # 任意想要接收数据的命令码，这里是雷达标记进度的命令码0x020E
    vulnerability_cmd_id = [0x02, 0x0E]  # 双倍易伤次数和触发状态
    target_cmd_id = [0x01, 0x05]  # 飞镖目标
    buffer = b''  # 初始化缓冲区
    while True:
        # 从串口读取数据
        received_data = ser1.read_all()  # 读取一秒内收到的所有串口数据
        # 将读取到的数据添加到缓冲区中
        buffer += received_data

        # 查找帧头（SOF）的位置
        sof_index = buffer.find(b'\xA5')

        while sof_index != -1:
            # 如果找到帧头，尝试解析数据包
            if len(buffer) >= sof_index + 5:  # 至少需要5字节才能解析帧头
                # 从帧头开始解析数据包
                packet_data = buffer[sof_index:]

                # 查找下一个帧头的位置
                next_sof_index = packet_data.find(b'\xA5', 1)

                if next_sof_index != -1:
                    # 如果找到下一个帧头，说明当前帧头到下一个帧头之间是一个完整的数据包
                    packet_data = packet_data[:next_sof_index]
                    # print(packet_data)
                else:
                    # 如果没找到下一个帧头，说明当前帧头到末尾不是一个完整的数据包
                    break

                # 解析数据包
                progress_result = receive_packet(packet_data, progress_cmd_id,
                                                 info=False)  # 解析单个数据包，cmd_id为0x020E,不输出日志
                vulnerability_result = receive_packet(packet_data, vulnerability_cmd_id, info=False)
                target_result = receive_packet(packet_data, target_cmd_id, info=False)
                # 更新裁判系统数据，标记进度、易伤、飞镖目标
                if progress_result is not None:
                    received_cmd_id1, received_data1, received_seq1 = progress_result
                    progress_list = list(received_data1)
                    if state == 'R':
                        mark_progress['B1'] = progress_list[0]
                        mark_progress['B2'] = progress_list[1]
                        mark_progress['B3'] = progress_list[2]
                        mark_progress['B4'] = progress_list[3]
                        mark_progress['B5'] = progress_list[4]
                        mark_progress['B7'] = progress_list[5]
                        print("========== Serial Receive ==========")
                        print("mark_progress_B1: ", mark_progress['B1'])
                        print("mark_progress_B2: ", mark_progress['B2'])
                        print("mark_progress_B3: ", mark_progress['B3'])
                        print("mark_progress_B4: ", mark_progress['B4'])
                        print("mark_progress_B5: ", mark_progress['B5'])
                        print("mark_progress_B7: ", mark_progress['B7'])
                    else:
                        mark_progress['R1'] = progress_list[0]
                        mark_progress['R2'] = progress_list[1]
                        mark_progress['R3'] = progress_list[2]
                        mark_progress['R4'] = progress_list[3]
                        mark_progress['R5'] = progress_list[4]
                        mark_progress['R7'] = progress_list[5]
                        print("mark_progress_R1: ", mark_progress['R1'])
                        print("mark_progress_R2: ", mark_progress['R2'])
                        print("mark_progress_R3: ", mark_progress['R3'])
                        print("mark_progress_R4: ", mark_progress['R4'])
                        print("mark_progress_R5: ", mark_progress['R5'])
                        print("mark_progress_R7: ", mark_progress['R7'])
                if vulnerability_result is not None:
                    received_cmd_id2, received_data2, received_seq2 = vulnerability_result
                    received_data2 = list(received_data2)[0]
                    double_vulnerability_chance, opponent_double_vulnerability = Radar_decision(received_data2)
                    print("double_vulnerability_chance: ", double_vulnerability_chance)
                if target_result is not None:
                    received_cmd_id3, received_data3, received_seq3 = target_result
                    target = (list(received_data3)[1] & 0b1100000) >> 5
                    print("target:", target)

                # 从缓冲区中移除已解析的数据包
                buffer = buffer[sof_index + len(packet_data):]

                # 继续寻找下一个帧头的位置
                sof_index = buffer.find(b'\xA5')

            else:
                # 缓冲区中的数据不足以解析帧头，继续读取串口数据
                break
        time.sleep(0.5)


# 创建机器人坐标滤波器
filter = Filter(window_size=3, max_inactive_time=2)

# 加载模型，实例化机器人检测器和装甲板检测器
# weights_path = 'models/car.onnx'  # 建议把模型转换成TRT的engine模型，推理速度提升10倍，转换方式看README
# weights_path_next = 'models/armor.onnx'
weights_path = 'models/car.engine'
weights_path_next = 'models/armor.engine'
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv5Detector(weights_path_next, data='yaml/armor.yaml', conf_thres=0.50, iou_thres=0.2,
                               max_det=1,
                               ui=True)

ser1 = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # 串口
# 图像测试模式（获取图像根据自己的设备，在）
camera_mode = 'video'  # 'test':图片测试, 'video':视频测试, 'hik':海康相机, 'galaxy':大恒相机, 'usb':USB相机


# 串口接收线程
thread_receive = threading.Thread(target=ser_receive, daemon=True)
thread_receive.start()

# 串口发送线程
thread_list = threading.Thread(target=ser_send, daemon=True)
thread_list.start()

camera_image = None

if camera_mode == 'test':
    camera_image = cv2.imread('images/test_image.jpg')
    # camera_image = cv2.imread('images/test03.png')
elif camera_mode == 'usb':
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'galaxy':
    thread_camera = threading.Thread(target=gx_camera_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'hik':
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'video':
    thread_camera = threading.Thread(target=video_test_get, daemon=True)
    thread_camera.start()

while camera_image is None:
    print("等待图像。。。")
    time.sleep(0.5)

# 获取相机图像的画幅，限制点不超限
img0 = camera_image.copy()
img_y = img0.shape[0]
img_x = img0.shape[1]
print(img0.shape)

while True:
    # 刷新裁判系统信息UI图像
    information_ui_show = information_ui.copy()
    map = map_backup.copy()
    det_time = 0
    img0 = camera_image.copy()
    ts = time.time()

    # 第一层神经网络识别
    result0 = detector.predict(img0)
    det_time += 1
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # 存储第一次检测结果和区域
            # ROI出机器人区域
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            # 第二层神经网络识别
            result_n = detector_next.predict(cropped_img)
            det_time += 1
            if result_n:
                # 叠加第二次检测结果到原图的对应位置
                img0[top:top + h, left:left + w] = cropped_img

                for detection1 in result_n:
                    cls, xywh, conf = detection1
                    if cls:  # 所有装甲板都处理，可选择屏蔽一些:
                        x, y, w, h = xywh
                        x = x + left
                        y = y + top

                        t1 = time.time()
                        # 原图中装甲板的中心下沿作为待仿射变化的点
                        camera_point = np.array([[[min(x + 0.5 * w, img_x), min(y + 1.5 * h, img_y)]]],
                                                dtype=np.float32)
                        # 低到高依次仿射变化
                        # 先套用地面层仿射变化矩阵
                        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                        # 限制转换后的点在地图范围内
                        x_c = max(int(mapped_point[0][0][0]), 0)
                        y_c = max(int(mapped_point[0][0][1]), 0)
                        x_c = min(x_c, width)
                        y_c = min(y_c, height)
                        color = mask_image[y_c, x_c]  # 通过掩码图像，获取地面层的颜色：黑（0，0，0）

                        if color[0] == color[1] == color[2] == 0:
                            X_M = x_c
                            Y_M = y_c
                            # Z_M = 0
                            color_flag = filter.add_data(cls, X_M, Y_M, conf)
                        else:
                            # 不满足则继续套用R型高地层仿射变换矩阵
                            mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_height_r)
                            # 限制转换后的点在地图范围内
                            x_c = max(int(mapped_point[0][0][0]), 0)
                            y_c = max(int(mapped_point[0][0][1]), 0)
                            x_c = min(x_c, width)
                            y_c = min(y_c, height)
                            color = mask_image[y_c, x_c]  # 通过掩码图像，获取R型高地层的颜色：绿（0，255，0）
                            if color[1] > color[2] and color[1] > color[0]:
                                X_M = x_c
                                Y_M = y_c
                                # Z_M = 400
                                color_flag = filter.add_data(cls, X_M, Y_M, conf)
                            else:
                                # 不满足则继续套用环形高地层仿射变换矩阵
                                mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_height_g)
                                # 限制转换后的点在地图范围内
                                x_c = max(int(mapped_point[0][0][0]), 0)
                                y_c = max(int(mapped_point[0][0][1]), 0)
                                x_c = min(x_c, width)
                                y_c = min(y_c, height)
                                color = mask_image[y_c, x_c]  # 通过掩码图像，获取环型高地层的颜色：蓝（255，0，0）
                                if color[0] > color[2] and color[0] > color[1]:
                                    X_M = x_c
                                    Y_M = y_c
                                    # Z_M = 600
                                    color_flag = filter.add_data(cls, X_M, Y_M, conf)
                        cls0 = cls
                        if color_flag == 1:
                            if cls.startswith('R'):
                                new_cls = cls.replace('R', 'B') 
                            if cls.startswith('B'):
                                new_cls = cls.replace('B', 'R')
                            cls0 = new_cls
                        hide = hide_mask[y_c, x_c]
                        # print('x_c: ', x_c)
                        # print('y_c: ', y_c)
                        # print(hide)
                        if hide[0] > hide[1] and hide[0] > hide[2]:
                            guess_index[cls0] = 1
                        if hide[1] > hide[2] and hide[1] > hide[0]:
                            guess_index[cls0] = 2
                        if hide[2] > hide[1] and hide[2] > hide[0]:
                            guess_index[cls0] = 3
                        else:
                            guess_index[cls0] = 0


    # 获取所有识别到的机器人坐标
    all_filter_data = filter.get_all_data()
    # print(all_filter_data_name)
    print('guess index:')
    print(guess_index)
    print('guess value')
    print(guess_value)
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)
                else:
                    color_m = (255, 0, 0)
                if state == 'R':
                    filtered_xyz = (2800 - xyxy[1], xyxy[0])  # 缩放坐标到地图图像
                else:
                    filtered_xyz = (xyxy[1], 1500 - xyxy[0])  # 缩放坐标到地图图像
                # 只绘制敌方阵营的机器人（这里不会绘制盲区预测的机器人）
                # if name[0] != state:
                cv2.circle(map, (int(filtered_xyz[0]), int(filtered_xyz[1])), 20, color_m, -1)  # 绘制圆
                cv2.putText(map, str(name),
                            (int(filtered_xyz[0]) - 5, int(filtered_xyz[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                ser_x = int(filtered_xyz[0]) * 10 / 1000
                ser_y = int(1500 - filtered_xyz[1]) * 10 / 1000
                cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                            (int(filtered_xyz[0]) - 100, int(filtered_xyz[1]) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    te = time.time()
    t_p = te - ts
    print("fps:",1 / t_p)  # 打印帧率
    # 绘制UI
    # _ = draw_information_ui(progress_list, state, information_ui_show)
    # cv2.putText(information_ui_show, "vulnerability_chances: " + str(double_vulnerability_chance),
    #             (10, 350),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(information_ui_show, "vulnerability_Triggering: " + str(opponent_double_vulnerability),
    #             (10, 400),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.imshow('information_ui', information_ui_show)
    map_show = cv2.resize(map, (1200, 640))
    cv2.imshow('map', map_show)
    img0 = cv2.resize(img0, (1300, 900))
    cv2.imshow('img', img0)

    key = cv2.waitKey(1)
