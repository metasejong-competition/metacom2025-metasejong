import numpy as np
import carb
import os
from pathlib import Path
import traceback

from scipy.spatial.transform import Rotation as R
import omni.graph.core as og
from omni.isaac.core_nodes import BaseResetNode
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
from typing import Optional
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.scenes import Scene
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_rot_matrix

from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.usd import get_world_transform_matrix

# rclpy 관련 import
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SCALE_FACTOR = 0.05

# Fallback joint values (12 joints) in proper order.
FALLBACK_JOINT_POSITIONS = np.array(
    [
        -1.8336,  # joint_3
        0.0,  # joint_4
        -0.5505,  # joint_2
        -1.117,  # joint_5
        0.0,  # joint_1
        1.5604,  # joint_6
        0,
        0,
        0,
        0,
        0,
        0,  # front_left_wheel_joint, front_right_wheel_joint, ...
    ]
)


# rclpy Publisher Node for IK failure notifications
class IKFailurePublisher(Node):
    def __init__(self):
        super().__init__("ik_failure_publisher")
        self.publisher_ = self.create_publisher(String, "ik_failure_topic", 10)

    def publish_failure(self, msg: str):
        message = String()
        message.data = msg
        self.publisher_.publish(message)
        self.get_logger().info(f"Published IK failure: {msg}")


global_ik_failure_publisher = None


def get_ik_failure_publisher():
    """
    rclpy 노드를 전역으로 단 한 번만 생성하기 위한 함수.
    이미 생성된 경우 그대로 반환, 없으면 init 후 생성.
    """
    global global_ik_failure_publisher
    if global_ik_failure_publisher is None:
        try:
            rclpy.init(args=None)
        except RuntimeError as e:
            # "Context.init() must only be called once" 오류는 이미 초기화되었음을 의미하므로 무시
            if "Context.init() must only be called once" in str(e):
                pass
            else:
                raise e
        global_ik_failure_publisher = IKFailurePublisher()
    return global_ik_failure_publisher


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(
        self,
        robot_articulation: Articulation,
        robot_description_path: str,
        urdf_path: str,
        end_effector_frame_name: Optional[str] = None,
    ) -> None:
        self._kinematics = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
        )
        end_effector_frame_name = "end_effector_link"
        ArticulationKinematicsSolver.__init__(
            self, robot_articulation, self._kinematics, end_effector_frame_name
        )


def convert_quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """
    q: [x, y, z, w] -> [w, x, y, z]
    """
    return np.array([q[3], q[0], q[1], q[2]])


def quat_wxyz_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    입력 쿼터니언 q가 [w, x, y, z] 순서라고 가정하고,
    이를 [x, y, z, w] 순서로 변환하여 회전 행렬을 반환합니다.
    """
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(q_xyzw).as_matrix()


def remove_scale(T):
    R = T[:3, :3]
    t = T[:3, 3]  # T[3, :3]는 column-major 기준일 때만!

    # 열 벡터 각각의 norm으로 스케일 제거
    scale_x = np.linalg.norm(R[:, 0])
    scale_y = np.linalg.norm(R[:, 1])
    scale_z = np.linalg.norm(R[:, 2])

    R[:, 0] /= scale_x
    R[:, 1] /= scale_y
    R[:, 2] /= scale_z

    T_noscale = np.eye(4)
    T_noscale[:3, :3] = R
    T_noscale[:3, 3] = t
    return T_noscale


def gf_matrix_to_numpy(mat):
    """Gf.Matrix4d 같은 column-major matrix를 row-major numpy 4x4로 변환"""
    return np.array(mat).reshape(4, 4).T  # Transpose 필요!!

def get_translate_from_prim(
    translation_from_source: np.ndarray,
    source_prim,
    target_info: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:

    # prim 경로
    base_link_path = "/world/metacombot/scout_v2_base/base_link"
    manipulator_link_path = "/world/metacombot/metacom_robot/kinova_robot_manipulator/base_link"

    # prim 불러오기
    base_link_prim = get_prim_at_path(base_link_path)
    manipulator_link_prim = get_prim_at_path(manipulator_link_path)

    # 월드 트랜스폼 가져오고 numpy로 변환 + transpose
    T_world_base = gf_matrix_to_numpy(get_world_transform_matrix(base_link_prim))
    T_world_manip = gf_matrix_to_numpy(get_world_transform_matrix(manipulator_link_prim))

    # base 쪽 스케일 제거
    T_world_base_noscale = remove_scale(T_world_base)

    # manipulator 쪽은 그대로 사용
    T_world_manip_noscale = T_world_manip

    # 상대 변환
    T_robot2world = np.linalg.inv(T_world_base_noscale) @ T_world_manip_noscale

    cur_target_object_pos, current_target_object_ori = target_info
    # target의 회전도 동일한 컨벤션으로 변환
    R_target = quat_wxyz_to_rot_matrix(current_target_object_ori)
    T_obj2world = np.eye(4, dtype=float)
    T_obj2world[0:3, 0:3] = R_target * SCALE_FACTOR
    T_obj2world[0:3, 3] = cur_target_object_pos

    # draw = _debug_draw.acquire_debug_draw_interface()
    # draw.clear_points()
    # draw.draw_points(
    #     [robot_pos + translation_rotated],
    #     [(1.0, 0.0, 0.0, 1.0)],
    #     [100],
    # )
    # draw.draw_points([cur_target_object_pos], [(0.0, 1.0, 0.0, 1.0)], [100])

    T_robot2obj = np.linalg.inv(T_robot2world) @ T_obj2world

    target_pos = np.pad(translation_from_source, (0, 1), constant_values=1)
    target_cube_relative_coordinates = (target_pos @ np.transpose(T_robot2obj))[0:3]

    return target_cube_relative_coordinates


class OgnPickNPlaceInit(BaseResetNode):
    """
    BaseResetNode를 상속하여, 로봇 초기화 및 상태 유지.
    """

    def __init__(self):
        self.initialized = None
        self.robot_prim_path = None
        self.ee_prim_path = None
        self.robot_name = None
        self.ee_name = None
        self.manipulator = None
        self.robot_controller = None
        self.robot_description_path = None
        self.urdf_path = None

        self.waypoints = None
        self.current_waypoint_idx = 0
        self.previous_actions = None
        self.task = "Idle"
        self.timestamp = 0
        self.ik_fail_count = 0

        super().__init__(initialize=False)

    def initialize_scene(self):
        self.scene = Scene()
        self.manipulator = SingleManipulator(
            prim_path=self.robot_prim_path,
            name=self.robot_name,
            end_effector_prim_name=self.ee_name,
        )
        self.scene.add(self.manipulator)
        self.manipulator.initialize()

        self.robot = self.scene.get_object(self.robot_name)
        self.robot_controller = KinematicsSolver(
            robot_articulation=self.robot,
            robot_description_path=self.robot_description_path,
            urdf_path=self.urdf_path,
        )
        self.articulation_controller = self.robot.get_articulation_controller()

        self.initialized = True
        self.timestamp = 0
        self.previous_actions = None
        self.waypoints = None

    def custom_reset(self):
        self.robot_controller = None


def move_along_waypoints(
    state: OgnPickNPlaceInit,
    robot_xform_prim: XFormPrim,
    start_offset: np.ndarray,
    end_offset: np.ndarray,
    reference_info: tuple[np.ndarray, np.ndarray],
    orientation: np.ndarray,
    num_steps: int,
    next_task_name: str,
    db: og.Database,
) -> bool:
    """
    웨이포인트를 순차적으로 이동하는 로직.
    - 웨이포인트가 None이면 start_offset -> end_offset 구간을 num_steps등분.
    - IK 실패 횟수가 50회 이상이면 Done 상태로 전환.
    """

    if state.waypoints is None:
        start_point = np.round(
            get_translate_from_prim(start_offset, robot_xform_prim, reference_info),
            decimals=3,
        )
        end_point = np.round(
            get_translate_from_prim(end_offset, robot_xform_prim, reference_info),
            decimals=3,
        )
        state.waypoints = [
            start_point + (end_point - start_point) * i / (num_steps - 1)
            for i in range(num_steps)
        ]
        state.current_waypoint_idx = 0
        print(f"  -> Waypoints initialized: {state.waypoints}")

    cur_target = state.waypoints[state.current_waypoint_idx]
    actions, succ = state.robot_controller.compute_inverse_kinematics(
        target_position=np.array(cur_target),
        target_orientation=orientation,
    )
    print(f"  -> IK success={succ}")

    if succ:
        state.ik_fail_count = 0
        state.articulation_controller.apply_action(actions)
        state.current_waypoint_idx += 1
        print(f"  -> Reached waypoint! Next idx={state.current_waypoint_idx}")

        if state.current_waypoint_idx >= len(state.waypoints):
            state.task = next_task_name
            state.waypoints = None
            state.current_waypoint_idx = 0
            return True

    else:
        state.ik_fail_count += 1
        carb.log_warn(
            f"IK did not converge for target {cur_target}, failure count: {state.ik_fail_count}"
        )
        db.outputs.pick_and_place_command = True

        if state.ik_fail_count >= 50:
            print("IK failed 50 times in current state. Transitioning to Done state.")
            publisher = get_ik_failure_publisher()
            publisher.publish_failure(
                "IK failed 50 times in current state. Transitioning to Done state."
            )
            state.task = "Done"
            state.ik_fail_count = 0
            state.waypoints = None
            state.current_waypoint_idx = 0
            return True

        return False

    return False


class OgnPickandPlace:
    """
    OmniGraph Node:
    - setup(db): 노드 생성 시 한 번 호출
    - compute(db): 매 프레임(또는 tick)마다 호출
    - cleanup(db): 노드 삭제/Reset 시 호출
    """

    @staticmethod
    def internal_state():
        return OgnPickNPlaceInit()

    @staticmethod
    def setup(db: og.Database):
        """
        노드 생성 시 또는 'Reset' 버튼 시 호출됩니다.
        """
        pass

    @staticmethod
    def compute(db) -> bool:
        TOLERANCE = 1e-1
        state = db.internal_state
        ee_down_orientation = np.array(
            [0, 1, 0, 0]
        )  # 엔드이펙터가 아래로 향하도록 설정
        APART_TRANSLATE = 7
        CENTER_TRANSLATE = 2.7
        num_steps = 40

        try:
            if not state.initialized:
                if db.inputs.robot_prim_path == "":
                    carb.log_warn("robot prim path is not set")
                    return False

                SELECTED_ROBOT = "kinova_robot"
                state.robot_prim_path = db.inputs.robot_prim_path
                state.ee_name = "end_effector_link"
                state.robot_name = SELECTED_ROBOT

                robot_cfg_path = Path(__file__).parent.parent / "robot" / SELECTED_ROBOT
                state.robot_description_path = os.path.join(
                    robot_cfg_path, SELECTED_ROBOT + "_descriptor.yaml"
                )
                state.urdf_path = os.path.join(robot_cfg_path, SELECTED_ROBOT + ".urdf")

                state.task = "Idle"
                state.initialize_scene()
                carb.log_info("Robot Initialized")

            current_target_object_ori = db.inputs.grasping_point_ori
            cur_target_object_pos = db.inputs.grasping_point_pos
            current_target_object_ori = convert_quat_xyzw_to_wxyz(
                np.array(current_target_object_ori)
            )
            current_target_obj_info = (cur_target_object_pos, current_target_object_ori)

            object_target_ori = db.inputs.placement_point_ori
            object_target_pos = db.inputs.placement_point_pos
            object_target_ori = convert_quat_xyzw_to_wxyz(np.array(object_target_ori))
            placement_of_target_obj_info = (object_target_pos, object_target_ori)

            robot_xform_prim = XFormPrim(state.robot_prim_path)
            print(f"\n[RobotPickNPlace] Current State: {state.task}")

            # 상태머신
            if state.task == "Idle":
                print("  -> Transition to MoveAboveCube")
                db.outputs.gripper_grasp_command = False
                state.task = "MoveAboveCube"

            elif state.task == "MoveAboveCube":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True

                translation_from_source = np.array([0, 0, -APART_TRANSLATE])
                target_cube_relative_coordinates = get_translate_from_prim(
                    translation_from_source, robot_xform_prim, current_target_obj_info
                )
                actions, succ = state.robot_controller.compute_inverse_kinematics(
                    target_position=target_cube_relative_coordinates,
                    target_orientation=current_target_object_ori,
                )
                print(f"  -> IK Result: succ={succ}")
                if succ:
                    state.ik_fail_count = 0
                    state.previous_actions = actions
                    state.articulation_controller.apply_action(actions)
                    print("  -> Applying IK action, moving to ReachDown")
                else:
                    state.ik_fail_count += 1
                    db.outputs.pick_and_place_command = True
                    carb.log_warn(
                        f"Failed to move above cube, failure count: {state.ik_fail_count}"
                    )
                    if state.ik_fail_count >= 50:
                        print(
                            "IK failed 50 times in MoveAboveCube. Transitioning to Done state."
                        )
                        publisher = get_ik_failure_publisher()
                        publisher.publish_failure(
                            "IK failed 50 times in MoveAboveCube."
                        )
                        state.task = "Done"
                        state.ik_fail_count = 0
                        return True
                    print(traceback.format_exc())

                joint_pos_dict = state.robot.get_joint_positions()
                if isinstance(joint_pos_dict, dict):
                    current_joint_positions = np.array(
                        joint_pos_dict.joint_positions
                    ).flatten()
                elif isinstance(joint_pos_dict, (list, np.ndarray)):
                    current_joint_positions = np.array(joint_pos_dict).flatten()
                else:
                    raise TypeError(f"Unexpected type: {type(joint_pos_dict)}")

                robot_dof_names = state.robot.dof_names
                target_dof_names = [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                ]
                indices = [
                    robot_dof_names.index(name)
                    for name in target_dof_names
                    if name in robot_dof_names
                ]
                joint_pos_dict = joint_pos_dict[indices]
                prev_actions = state.previous_actions.joint_positions

                min_length = min(len(current_joint_positions), len(prev_actions))
                current_joint_positions = current_joint_positions[indices]
                prev_actions = prev_actions[:min_length]

                if np.allclose(current_joint_positions, prev_actions, atol=TOLERANCE):
                    print(
                        f"  -> Joint positions match previous action (tolerance {TOLERANCE}), transitioning to ReachDown"
                    )
                    state.task = "ReachDown"

            elif state.task == "ReachDown":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                done = move_along_waypoints(
                    state,
                    robot_xform_prim,
                    start_offset=np.array([0, 0, -APART_TRANSLATE]),
                    end_offset=np.array([0, 0, -CENTER_TRANSLATE]),
                    reference_info=current_target_obj_info,
                    orientation=current_target_object_ori,
                    num_steps=num_steps,
                    next_task_name="Pick",
                    db=db,
                )
                # 완료되면 Pick 상태로 전환

            elif state.task == "Pick":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                print("  -> In Pick state: closing gripper")
                db.outputs.gripper_grasp_command = True
                print("  -> Transition to PickUpObject")
                state.task = "PickUpObject"

            elif state.task == "PickUpObject":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                done = move_along_waypoints(
                    state,
                    robot_xform_prim,
                    start_offset=np.array([0, 0, -CENTER_TRANSLATE]),
                    end_offset=np.array([0, 0, -APART_TRANSLATE]),
                    reference_info=current_target_obj_info,
                    orientation=current_target_object_ori,
                    num_steps=num_steps,
                    next_task_name="MoveToTargetAbove",
                    db=db,
                )
                if not done:
                    db.outputs.pick_and_place_command = True
                    state.ik_fail_count += 1
                    if state.ik_fail_count >= 50:
                        print(
                            "IK failed 50 times in PickUpObject. Transitioning to Done state."
                        )
                        publisher = get_ik_failure_publisher()
                        publisher.publish_failure("IK failed 50 times in PickUpObject.")
                        state.task = "Done"
                        state.ik_fail_count = 0
                        return True

            elif state.task == "MoveToTargetAbove":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                translation_from_source = np.array([0, 0, -APART_TRANSLATE])
                target_cube_relative_coordinates = get_translate_from_prim(
                    translation_from_source,
                    robot_xform_prim,
                    placement_of_target_obj_info,
                )
                actions, succ = state.robot_controller.compute_inverse_kinematics(
                    target_position=target_cube_relative_coordinates,
                    target_orientation=ee_down_orientation,
                )
                print(f"  -> IK Result: succ={succ}")
                if succ:
                    state.ik_fail_count = 0
                    state.previous_actions = actions
                    state.articulation_controller.apply_action(actions)
                    print("  -> Applying IK action, moving to ReachDown")
                else:
                    state.ik_fail_count += 1
                    db.outputs.pick_and_place_command = True
                    carb.log_warn(
                        f"Failed to move above target, failure count: {state.ik_fail_count}"
                    )
                    if state.ik_fail_count >= 50:
                        print(
                            "IK failed 50 times in MoveToTargetAbove. Transitioning to Done state."
                        )
                        publisher = get_ik_failure_publisher()
                        publisher.publish_failure(
                            "IK failed 50 times in MoveToTargetAbove."
                        )
                        state.task = "Done"
                        state.ik_fail_count = 0
                        return True
                    print(traceback.format_exc())

                joint_pos_dict = state.robot.get_joint_positions()
                if isinstance(joint_pos_dict, dict):
                    current_joint_positions = np.array(
                        joint_pos_dict.joint_positions
                    ).flatten()
                elif isinstance(joint_pos_dict, (list, np.ndarray)):
                    current_joint_positions = np.array(joint_pos_dict).flatten()
                else:
                    raise TypeError(f"Unexpected type: {type(joint_pos_dict)}")

                robot_dof_names = state.robot.dof_names
                target_dof_names = [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                ]
                indices = [
                    robot_dof_names.index(name)
                    for name in target_dof_names
                    if name in robot_dof_names
                ]
                joint_pos_dict = joint_pos_dict[indices]
                prev_actions = state.previous_actions.joint_positions

                min_length = min(len(current_joint_positions), len(prev_actions))
                current_joint_positions = current_joint_positions[indices]
                prev_actions = prev_actions[:min_length]

                if np.allclose(current_joint_positions, prev_actions, atol=TOLERANCE):
                    print(
                        f"  -> Joint positions match previous action (tolerance {TOLERANCE}), transitioning to ReachDownTarget"
                    )
                    state.task = "ReachDownTarget"

            elif state.task == "ReachDownTarget":
                db.outputs.pick_and_place_command = True
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                done = move_along_waypoints(
                    state,
                    robot_xform_prim,
                    start_offset=np.array([0, 0, -APART_TRANSLATE]),
                    end_offset=np.array([0, 0, -CENTER_TRANSLATE]),
                    reference_info=placement_of_target_obj_info,
                    orientation=object_target_ori,
                    num_steps=num_steps,
                    next_task_name="Place",
                    db=db,
                )

            elif state.task == "Place":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                print("  -> In Place state: opening gripper")
                db.outputs.gripper_grasp_command = False
                print("  -> Transition to MoveAwayFromTarget")
                state.task = "MoveAwayFromTarget"

            elif state.task == "MoveAwayFromTarget":
                db.outputs.execute = og.ExecutionAttributeState.ENABLED
                db.outputs.pick_and_place_command = True
                done = move_along_waypoints(
                    state,
                    robot_xform_prim,
                    start_offset=np.array([0, 0, -CENTER_TRANSLATE]),
                    end_offset=np.array([0, 0, -APART_TRANSLATE]),
                    reference_info=placement_of_target_obj_info,
                    orientation=object_target_ori,
                    num_steps=num_steps,
                    next_task_name="Done",
                    db=db,
                )

            elif state.task == "Done":
                db.outputs.pick_and_place_command = True
                print("  -> In Done state: applying fallback action")
                fallback_action = ArticulationAction(
                    joint_positions=FALLBACK_JOINT_POSITIONS
                )
                state.articulation_controller.apply_action(fallback_action)

                current_joint_positions = state.robot.get_joint_positions()
                if isinstance(current_joint_positions, dict):
                    current_joint_positions = np.array(
                        current_joint_positions.joint_positions
                    ).flatten()
                elif isinstance(current_joint_positions, (list, np.ndarray)):
                    current_joint_positions = np.array(
                        current_joint_positions
                    ).flatten()
                else:
                    raise TypeError(f"Unexpected type: {type(current_joint_positions)}")

                # L2 norm으로 오차 계산
                error_norm = np.linalg.norm(
                    current_joint_positions[:6] - FALLBACK_JOINT_POSITIONS[:6]
                )
                print(f"Error norm: {error_norm}")
                if error_norm <= TOLERANCE:
                    db.outputs.pick_and_place_command = False
                    print(
                        "  -> Fallback joint positions reached within tolerance. Transitioning to Idle."
                    )
                    state.task = "Idle"

            elif state.task == "Idle":
                print("Idle")

        except Exception as error:
            db.outputs.pick_and_place_command = False
            db.log_warn(str(error))
            print(traceback.format_exc())
            return False

        return True

    @staticmethod
    def cleanup(db: og.Database):
        """
        OmniGraph 노드가 삭제되거나 Reset 버튼이 눌렸을 때 호출됩니다.
        여기서 ROS 노드를 정리(destroy_node)하고 shutdown()을 호출하여
        [publisher pointer is invalid] 에러를 줄일 수 있습니다.
        """
        global global_ik_failure_publisher
        if global_ik_failure_publisher is not None:
            # ROS 노드 destroy
            global_ik_failure_publisher.destroy_node()
            global_ik_failure_publisher = None
        # rclpy shutdown
        try:
            rclpy.shutdown()
        except Exception as e:
            # 이미 shutdown 되었거나, 초기화 안 된 상태면 무시
            pass
        carb.log_info("ROS Cleanup done. rclpy.shutdown() called.")
