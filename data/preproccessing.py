import numpy as np
import cv2, os
from tqdm import tqdm
import shutil
from os.path import join as osp


# filename = "0008"
sharp_data = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/PhoneCraft2/original/Sharp/images"
save_data = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/PhoneCraft2/FakeBlur_opencv"
num_blur = 5
delta = int(num_blur / 2)

sequence = sorted(os.listdir(sharp_data))
for cate in sequence:
    framelist = sorted(os.listdir(osp(sharp_data, cate)))
    for idx, each in enumerate(tqdm(framelist[::2])):
        idx *= 2
        if idx < delta or idx >= len(framelist) - delta: continue
        # files = sorted(os.listdir(os.path.join(type_data, each)))
        # for f in tqdm(files):
            # frames = sorted(os.listdir(os.path.join(type_data, each, f)))
        starting = True
        prev_frame = np.uint8([250])
        average_frame = None

        num_frame = 0
        # breakpoint()
        for frame_name in framelist[idx - delta:idx + delta + 1]:
            # if "blur" in eachf: continue
            frame = cv2.imread(osp(sharp_data, cate, frame_name)).astype(float)
            
            if starting==True:
                prev_frame = frame
                starting = False
                average_frame = frame.copy()
                # average_frame1 = frame.copy()
                num_frame = 1
            else:
                N = 10
                tmp = prev_frame
                for i in range(1,N + 1):
                    weight = i/N
                
                    #get the blended frames in between
                    # mid_frame = cv2.addWeighted(prev_frame,weight,frame,1-weight,0)     
                    # average_frame1 += mid_frame.astype(float)

                    mid_frame_ = cv2.addWeighted(tmp,weight,frame,1-weight,0)     
                    # # cv2.imshow('prev',prev_frame)
                    # # cv2.imshow('mid',mid_frame)
                    # # cv2.imshow('cur',frame)
                    average_frame += mid_frame_.astype(float)

                    num_frame += 1
                    tmp = mid_frame_
                    # cv2.waitKey(0)
                average_frame += frame.astype(float)
                # average_frame1 += frame.astype(float)
                num_frame += 1

                #cv2.imshow('frame',frame)     
                prev_frame = frame
                
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        average_frame /= float(num_frame)
        # print(type_data, each, f)
        cv2.imwrite(os.path.join(save_data, "blur", cate + "_" + each), average_frame.astype("uint8"))
        shutil.copyfile(os.path.join(sharp_data, cate, each), os.path.join(save_data, "sharp", cate + "_" + each))
        # average_frame1 /= float(num_frame)
# cv2.imshow('average',average_frame.astype("uint8"))
# cv2.imshow('average1',average_frame1.astype("uint8"))
# blur = cv2.imread(filename + "/im_blur.png") 
# cv2.imshow('blur',blur)
# cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
