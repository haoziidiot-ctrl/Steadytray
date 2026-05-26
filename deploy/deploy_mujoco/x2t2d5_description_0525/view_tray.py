import mujoco
import mujoco.viewer
import os

xml = os.path.join(os.path.dirname(__file__), "x2_tray_hold.xml")
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.viewer.launch(m, d)
