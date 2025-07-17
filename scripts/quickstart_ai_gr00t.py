import time

import cv2
import numpy as np

from phosphobot.am import Gr00tN1
import httpx
from phosphobot.camera import AllCameras

host = "172.21.2.11"  # Change this to your server IP (this is the IP of the machine running the Gr00tN1 server using a GPU)
port = 8080

TASK_DESCRIPTION = (
    "pick up orange lego brock from left position to right position in black marker box."
)
print(f"{TASK_DESCRIPTION=}")
FPS = 30
PHOSPHOBOT_API_URL = "http://localhost:8020"

allcameras = AllCameras()
time.sleep(1)  # Wait for the cameras to initialize

for camera in allcameras.video_cameras:
    print(f"カメラID: {camera.camera_id}, タイプ: {camera.camera_type}, アクティブ: {camera.is_active}")
joint_ids = [1, 2, 3, 4, 5, 6]

if host == "YOUR_SERVER_IP":
    raise ValueError(
        "You need to change the host to the IP or URL of the machine running the Gr00tN1 server. It can be your local machine or a remote machine."
    )

while True:
    start_time = time.perf_counter()
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(224, 224)), # GR00Tは224,224
        allcameras.get_rgb_frame(camera_id=1, resize=(224, 224)),
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
    model = Gr00tN1(server_url=host, server_port=port)

    response = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()
    state = response["angles"]
    # Take a look at the experiment_cfg/metadata.json file in your Gr00t model and check the names of the images, states, and observations
    # You may need to adapt the obs JSON to match these names
    # The following JSON should work for one arm and 2 video cameras
    obs = {
        "video.image_cam_0": images[0], #cam_context
        "video.image_cam_1": images[1], #cam_wrist
        "state.arm_0": np.array(state[:6]).reshape(1, -1),  #single_arm # Reshape to (1, 5) for single arm
        # "state.gripper": np.array([state[5]]).reshape(1, -1),  # Reshape to (1, 1) for gripper
        "annotation.human.action.task_description": [TASK_DESCRIPTION],
    }

    actions = model.sample_actions(obs)

    for action in actions: 
        payload = {
            "angles": action.tolist(), 
            "unit": "rad", 
            "joints_ids": joint_ids,
        }
        httpx.post(
            url = f"{PHOSPHOBOT_API_URL}/joints/write", 
            json=payload,
            headers={"accept": "application/json"},
            params={"robot_id": 0},
        )
        # Wait to respect frequency control (30 Hz)
        elapsed_time = time.perf_counter() - start_time
        sleep_time = 1/FPS - elapsed_time        
        if sleep_time > 0:
            time.sleep(sleep_time)
