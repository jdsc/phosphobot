import time

import cv2
import numpy as np

from phosphobot.am import Gr00tN1
import httpx
from phosphobot.camera import AllCameras

host = "localhost"  # Change this to your server IP (this is the IP of the machine running the Gr00tN1 server using a GPU)
port = 8080

# Change this with your task description
TASK_DESCRIPTION = (
    "Pick up the orange lego brick from the table and put it in the black container."
)

# Connect to the phosphobot server, this is different from the server IP above
PHOSPHOBOT_API_URL = "http://localhost:8020"

allcameras = AllCameras()

# カメラ状態の確認
print("接続されているカメラIDs:", allcameras.camera_ids)
print("カメラ詳細情報:")
for camera in allcameras.video_cameras:
    print(f"カメラID: {camera.camera_id}, タイプ: {camera.camera_type}, アクティブ: {camera.is_active}")

time.sleep(3)  # カメラの初期化に十分な時間を与える

if host == "YOUR_SERVER_IP":
    raise ValueError(
        "You need to change the host to the IP or URL of the machine running the Gr00tN1 server. It can be your local machine or a remote machine."
    )

joint_ids = [1, 2, 3, 4, 5, 6]

while True:
    camera0_frame = allcameras.get_rgb_frame(camera_id=0, resize=(320, 240))
    camera1_frame = allcameras.get_rgb_frame(camera_id=1, resize=(320, 240))
    images = [
        camera0_frame,
        camera1_frame
    ]

    for i in range(0, len(images)):
        image = images[i]
        if image is None:
            print(f"Camera {i} is not available.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
        converted_array = np.expand_dims(image, axis=0)
        converted_array = converted_array.astype(np.uint8)
        images[i] = converted_array

    # Create the model, you might need to change the action keys based on your model, these can be found in the experiment_cfg/metadata.json file of your Gr00tN1 model
    action_keys = [
        "action.single_arm",
        "action.gripper",
    ]
    model = Gr00tN1(server_url=host, server_port=port, action_keys=action_keys)
    # ====================
    # debug
    ping_bool = model.client.ping()
    print(f"Ping successful: {ping_bool}")
    # ====================
    response = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()
    state = response["angles"]
    # Take a look at the experiment_cfg/metadata.json file in your Gr00t model and check the names of the images, states, and observations
    # You may need to adapt the obs JSON to match these names
    # The following JSON should work for one arm and 2 video cameras
    obs = {
        "video.cam_context": images[0],
        "video.cam_wrist": images[1],
        "state.single_arm": np.array(state[:5]).reshape(1, -1),  # Reshape to (1, 5) for single arm
        "state.gripper": np.array([state[5]]).reshape(1, -1),  # Reshape to (1, 1) for gripper
        "annotation.human.action.task_description": [TASK_DESCRIPTION],
    }

    actions = model.sample_actions(obs)
    for action in actions:
        payload = {
            "angles": action.tolist(),  # Assuming action is a list of actions
            "unit": "rad",  # or "rad" depending on your robot configuration
            "joints_ids": joint_ids,
        }
        httpx.post(
            url = f"{PHOSPHOBOT_API_URL}/joints/write",
            json=payload,
            headers={"accept": "application/json"},
            params={"robot_id": 1},
        )
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
