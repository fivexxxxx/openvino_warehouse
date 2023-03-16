from openvino.inference_engine import IECore
import cv2 as cv
import time
#人脸检测模型
model_xml = "model/face-detection-0202/FP32/face-detection-0202.xml"
model_bin = "model/face-detection-0202/FP32/face-detection-0202.bin"

# 加载头部姿态模型
em_xml = "model/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
em_bin = "model/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin"

def head_pose_estimation_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("model/emotions-recognition-retail-0003/FP32/222.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")
    em_net = ie.read_network(model=em_xml, weights=em_bin)
    em_input_blob = next(iter(em_net.input_info))
    #模型有3组输出
    head_it=iter(em_net.outputs)
    head_out_blob1=next(head_it)    #angle_p_fc
    head_out_blob2 = next(head_it)  #angle_r_fc
    head_out_blob3 = next(head_it)  #angle_y_fc

    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)
    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    #
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)  # 转换为c,h,w
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
        # print("infer time(ms): %.3f" %(inf_end*1000))
        ih, iw, ic = frame.shape
        res = res[out_blob]
        for obj in res[0][0]:
            if obj[2] > 0.75:
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax >= iw:
                    xmax = iw - 1
                if ymax >= ih:
                    ymax = ih - 1
                # 扣出人脸的区域
                roi = frame[ymin:ymax, xmin:xmax, :]
                roi_image = cv.resize(roi, (ew, eh))
                roi_image = roi_image.transpose(2, 0, 1)  # 转换为c,h,w
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_image]})
                # print(em_res)   # 下面是 2维的输出，1×1；则【0】【0】可取出其值
                angle_p_fc=em_res[head_out_blob1][0][0]
                angle_r_fc = em_res[head_out_blob2][0][0]
                angle_y_fc = em_res[head_out_blob3][0][0]
                head_pose=""
                if angle_p_fc>20 or angle_p_fc<-20:
                    head_pose+="pitch,"
                if angle_r_fc>20 or angle_r_fc<-20:
                    head_pose+="roll,"
                if angle_y_fc>20 or angle_y_fc<-20:
                    head_pose+="yaw"

                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame,"infer time(ms): %.3f, FPS:%.2f"%(inf_end*1000,1/inf_end),(50,50),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2,8)
                cv.putText(frame, head_pose, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 2.0,                           (0, 0, 255), 2, 8)

        cv.imshow("face+emotion detection", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=="__main__":

    head_pose_estimation_demo()