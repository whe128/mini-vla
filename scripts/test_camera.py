import mujoco
import mujoco.viewer
import cv2

MODEL_PATH = "arx_15/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


mujoco.mj_resetDataKeyframe(
    model,
    data,
    model.key("scene_home").id
)

# Camera
camera_id = model.camera("wrist_cam").id

# Renderer
renderer = mujoco.Renderer(
    model,
    height=480,
    width=640
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        # wrist camera
        renderer.update_scene(
            data,
            camera=camera_id
        )

        # MuJoCo -> RGB
        img = renderer.render()

        # RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # OpenCV show
        cv2.imshow("Wrist Camera", img)

        #
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        viewer.sync()

cv2.destroyAllWindows()
