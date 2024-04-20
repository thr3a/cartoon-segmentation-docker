import cv2
from PIL import Image
import numpy as np
from animeinsseg import AnimeInsSeg, AnimeInstances
import os
from tqdm import tqdm
import glob

# パラメータ
mask_thres = 0.3
instance_thres = 0.3
refine_kwargs = {"refine_method": "refinenet_isnet"}  # 使用しない場合はNoneを指定
# refine_kwargs = None

input_dir = "/images"
output_dir = "/images_output"
os.makedirs(output_dir, exist_ok=True)

file_list = glob.glob(os.path.join(input_dir, "*"), recursive=True)

print(len(file_list))

# AnimeInsSegの準備
ckpt = r"models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt"
net = AnimeInsSeg(ckpt, mask_thr=mask_thres, refine_kwargs=refine_kwargs)


def inference(img, filename):
    basename = os.path.basename(filename)
    output_filename = os.path.join(output_dir, basename)

    if os.path.exists(output_filename):
        return

    # 推論
    instances: AnimeInstances = net.infer(
        img, output_type="numpy", pred_score_thr=instance_thres
    )

    im_h, im_w = img.shape[:2]

    if type(instances.bboxes) != np.ndarray:
        return

    for ii, (xywh, mask) in enumerate(zip(instances.bboxes, instances.masks)):
        left = int(xywh[0])
        top = int(xywh[1])
        right = int(xywh[2]) + left
        bottom = int(xywh[3]) + top
        img_clopped = np.copy(img[top:bottom, left:right])

        # 背景を透過にするためにアルファチャンネルを追加
        img_clopped = cv2.cvtColor(img_clopped, cv2.COLOR_RGB2RGBA)

        # 背景を透過にする
        mask_clopped = mask[top:bottom, left:right]
        img_clopped[..., 3] = mask_clopped * 255

        basename = os.path.basename(filename)
        without_ext, ext = os.path.splitext(basename)
        output_filename = os.path.join(output_dir, f"{without_ext}_{str(ii)}.png")
        print("write:", output_filename)
        cv2.imwrite(output_filename, img_clopped)


for file in tqdm(file_list):
    try:
        if ".zip" in file:
            continue
        img = cv2.imread(file)
    except:
        continue

    inference(img, file)
