from openvino.inference_engine import IECore
import cv2 as cv
import time

def face_detection_demo():
    ie=IECore()
    for device in ie.available_devices:
        print(device)
    model_xml="model/face-detection-0202/FP32/face-detection-0202.xml"
    model_bin="model/face-detection-0202/FP32/face-detection-0202.bin"

    net=ie.read_network(model=model_xml,weights=model_bin)
    input_blob=next(iter(net.input_info))
    out_blob=next(iter(net.outputs))

    n,c,h,w=net.input_info[input_blob].input_data.shape
    print(n,c,h,w)  #1 3 384 384

    cap=cv.VideoCapture("model/face-detection-0202/FP32/111.mp4")
    exec_net=ie.load_network(network=net,device_name="CPU")
    #
    while True:
        ret,frame=cap.read()
        if ret is not True:
            break
        print(frame.shape)  #(360, 480, 3)
        image=cv.resize(frame,(w,h))
        print(image.shape)  #(384, 384, 3)
        image=image.transpose(2,0,1)#转换为c,h,w
        print(image.shape)
        inf_start=time.time()
        res=exec_net.infer(inputs={input_blob:[image]})
        inf_end=time.time()-inf_start
        print("infer time(ms): %.3f" %(inf_end*1000))
        print(frame.shape)
        ih,iw,ic=frame.shape
        print(res)
        res=res[out_blob]
        print("------------------------------")
        print(res)
        for obj in res[0][0]:
            if obj[2]>0.75:
                xmin=int(obj[3]*iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,255),2,8)
                cv.putText(frame,"infer time(ms): %.3f"%(inf_end*1000),(50,50),cv.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255),2,8)
        cv.imshow("detection",frame)
        c=cv.waitKey(1)
        if c==27:
            break
if __name__=="__main__":
    face_detection_demo()