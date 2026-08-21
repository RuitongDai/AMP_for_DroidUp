"""F2 robot MJCF, scene entity, and motion metadata."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

##
# MJCF and assets.
##

F2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "f2" / "xmls" / "f1_1_no_hand.xml"
)

assert F2_XML.exists(), f"F2 MJCF not found: {F2_XML}"


def get_spec() -> mujoco.MjSpec:
  # Empty spec.assets: MuJoCo resolves mesh files from disk (mjlab #873).
  return mujoco.MjSpec.from_file(str(F2_XML))



F2_ACTUATOR_LEGS_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)
F2_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",".*_hip_yaw_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=87.0,
  armature=0.01,
)
F2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=120.0,
  armature=0.01,
)

F2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=89.0,
  armature=0.01,
)

F2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=12.0,
  armature=0.01,
)


F2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_roll_joint"),
  stiffness=200.0,
  damping=5.0,
  effort_limit=87.0,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.86),
  joint_pos={
    # Legs.
    ".*_hip_pitch_joint": -0.1,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    ".*_ankle_roll_joint": 0.0,

    # Waist controlling torso_link.
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Entity config.
##

F2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    F2_ACTUATOR_LEGS_PITCH,
    F2_ACTUATOR_LEGS,
    F2_ACTUATOR_KNEE,
    F2_ACTUATOR_ANKLE_PITCH,
    F2_ACTUATOR_ANKLE_ROLL,
    F2_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_f2_robot_cfg() -> EntityCfg:
  """Get a fresh F2 robot configuration instance."""

  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=F2_ARTICULATION,
  )


F2_ACTION_SCALE: dict[str, float] = {}
for a in F2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    F2_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_f2_robot_cfg())
  viewer.launch(robot.spec.compile())
