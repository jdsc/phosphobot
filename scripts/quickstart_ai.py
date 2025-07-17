from phosphobot.camera import AllCameras
import httpx
from phosphobot.am import ACT

from collections import deque
import time
import numpy as np

import cv2

# カメラの画像を確認するデバッグ
IS_CAMERA_DEBUG = False
FPS = 30

# Connect to the phosphobot server
PHOSPHOBOT_API_URL = "http://localhost:8020"

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Instantiate the model
model = ACT(server_url="http://172.21.2.11", server_port=8080) # ローカルPCで推論するなら、urlはlocalhostにする

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json

joint_ids = [1, 2, 3, 4, 5, 6]

actions_queue: deque = deque([]) # phosphobot/am/act.pyを参考にしたqueue

while True:
    start_time = time.perf_counter()

    camera0_frame = allcameras.get_rgb_frame(camera_id=0, resize=(320, 240)) #公式は(240, 320)だが、縦長になる。cv2はH,Wなのでこれで横長。 #id0:wrist camera
    camera0_frame = cv2.cvtColor(camera0_frame, cv2.COLOR_RGB2BGR) # RGBに変換
    camera1_frame = allcameras.get_rgb_frame(camera_id=1, resize=(320, 240)) #1: 俯瞰カメラ
    # camera2_frame = allcameras.get_rgb_frame(camera_id=2, resize=(320, 240)) #2: PCカメラ

    # Get the robot state
    state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

    inputs = {
        "observation.state": np.array(state["angles"]),
        "observation.images.0": camera0_frame, # "observation.images.0"
        "observation.images.1": camera1_frame, #"observation.images.1"
    }

    # 各カメラの画像を表示する
    if IS_CAMERA_DEBUG:
        if inputs["observation.images.0"] is not None:
            cv2.imshow("Camera 0 Input", inputs["observation.images.0"])
        
        if inputs["observation.images.1"] is not None:
            cv2.imshow("Camera 1 Input", inputs["observation.images.1"])

        # キー入力を1ミリ秒待つ。'q'が押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if len(actions_queue) == 0:
        actions = model(inputs) #6要素のlist →修正後：6要素のlist×時系列分がネスト
        actions_queue.extend(actions)
    actions = actions_queue.popleft()

    # actions = actions[0].astype(np.float64).tolist()

    # Send the new joint postion to the robot
    inputs = {
        "angles": actions[0].astype(np.float64).tolist(),
        "unit": "rad",
        "joints_ids": joint_ids,
    }

    httpx.post(
        url = f"{PHOSPHOBOT_API_URL}/joints/write", 
        json=inputs,
        headers={"accept": "application/json"},
        params={"robot_id": 0}, # 1
    )

    # Wait to respect frequency control (30 Hz)
    # time.sleep(1 / 30)
    elapsed_time = time.perf_counter() - start_time
    # 目標周期から経過時間を引いて、待機すべき時間を計算
    sleep_time = 1/FPS - elapsed_time
    
    # 待機時間が0より大きい場合のみ待機する
    if sleep_time > 0:
        time.sleep(sleep_time)