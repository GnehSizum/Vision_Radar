import gxipy as gx
from PIL import Image
import sys
import numpy as np

def create_device(device_manager):
    print("")
    print("--------------------------------------------")
    print("- Daheng Galaxy Camera Py drive startup... -")
    print("--------------------------------------------")
    print("")
    print("Initializing......")
    print("")
 
    #创建设备
    dev_num, dev_info_list = device_manager.update_device_list() #枚举设备，即枚举所有可用的设备
    if dev_num == 0:
        print("Number of enumerated devices is 0")
        return
    else:
        print("--------------------------------------------")
        print("Successfully created device, Index is: %d" % dev_num)
        return dev_info_list

def open_device(cam):
    Width_set = 2048 # 设置分辨率宽
    Height_set = 1536 # 设置分辨率高

    #如果是黑白相机
    if cam.PixelColorFilter.is_implemented() is False: # is_implemented判断枚举型属性参数是否已实现
        print("该示例不支持黑白相机.")
        cam.close_device()
        return
    else:
        print("")
        print("--------------------------------------------")
        print("Successfully opened device")

    #设置宽和高
    cam.Width.set(Width_set)
    cam.Height.set(Height_set)

def start_acquisition(cam):
    framerate_set = 80 # 设置帧率
    #设置连续采集
    #cam.TriggerMode.set(gx.GxSwitchEntry.OFF) # 设置触发模式
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    #设置帧率
    cam.AcquisitionFrameRate.set(framerate_set)
    #开始数据采集
    print("")
    print("--------------------------------------------")
    print("Start acquisition...")
    print("")
    cam.stream_on()

def get_image(cam):
    raw_image = cam.data_stream[0].get_image() # 打开第0通道数据流
    if raw_image is None:
        print("获取彩色原始图像失败.")
        sys.exit()

    rgb_image = raw_image.convert("RGB") # 从彩色原始图像获取RGB图像
    if rgb_image is None:
        sys.exit()

    #rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)  # 实现图像增强

    numpy_image = rgb_image.get_numpy_array() # 从RGB图像数据创建numpy数组
    # numpy_image = np.asarray(rgb_image)
    if numpy_image is None:
        sys.exit()

    # img = Image.fromarray(numpy_image, 'RGB') # 展示获取的图像
    #img.show()
    return numpy_image

def stop_acquisition(cam):
    print("")
    print("--------------------------------------------")
    print("Stop acquisition...")
    cam.stream_off()

def close_device(cam):
    print("")
    print("--------------------------------------------")
    print("Successfully closed device!")
    cam.close_device()

def main():
    print("111")

if __name__ == "__main__":
    main()


    