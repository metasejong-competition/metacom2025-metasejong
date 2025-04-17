"""
This is the implementation of the OGN node defined in OgnPickandPlaceCommandReceiver.ogn
"""

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time
import uuid
from rclpy.executors import MultiThreadedExecutor

# 전역 변수 설정
subscriber_node = None
subscription = None
executor = None
last_message = None  # 마지막으로 수신된 메시지 (문자열)
command_start_time = None  # 메시지 수신 시각 (타이머 시작)
setup_done = False  # ROS2 구독 설정 여부


def listener_callback(msg):
    global last_message, command_start_time
    last_message = msg.data
    command_start_time = time.time()
    if subscriber_node is not None:
        subscriber_node.get_logger().info("Received: " + msg.data)


def ros_setup():
    """
    ROS2 초기화 및 노드, 구독자 설정을 한 번만 수행합니다.
    노드 이름에 UUID를 붙여 유일하게 생성하여 중복 등록 문제를 피합니다.
    """
    global subscriber_node, subscription, setup_done, executor
    if setup_done:
        return

    try:
        rclpy.init(args=None)
    except Exception as e:
        # 이미 초기화되어 있다면 예외 무시
        pass

    try:
        # 고유한 노드 이름 생성 (예: omnigraph_subscriber_ab12cd34)
        node_name = "omnigraph_subscriber_" + uuid.uuid4().hex[:8]
        print(f"DEBUG: Creating node with name: {node_name}")
        subscriber_node = Node(node_name)
        subscription = subscriber_node.create_subscription(
            String, "str", listener_callback, 10
        )
        print("DEBUG: Subscription created successfully")

        executor = MultiThreadedExecutor()
        executor.add_node(subscriber_node)
        threading.Thread(target=executor.spin, daemon=True).start()
        setup_done = True
        print("DEBUG: ros_setup completed successfully")
    except Exception as e:
        print(f"DEBUG: Exception during node setup: {str(e)}")
        raise


def ros_cleanup():
    """
    ROS2 노드 종료 및 정리
    """
    global subscriber_node, executor
    if subscriber_node is not None:
        subscriber_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


class OgnPickandPlaceCommandReceiver:
    """
    ROS2 'str' 토픽에서 String 메시지를 구독하여, 메시지 수신 후 1초 동안
    pickandplace_command(True)와 prim_name(메시지 내용)을 출력하고,
    이후에는 pickandplace_command는 False로 전환되지만 마지막 메시지(prim_name)는 유지합니다.
    """

    @staticmethod
    def compute(db) -> bool:
        global last_message, command_start_time
        try:
            # ROS2 구독 설정 (최초 1회)
            ros_setup()

            current_time = time.time()
            # 메시지 수신 후 1초 이내이면 pickandplace_command를 True로 출력
            if (
                command_start_time is not None
                and (current_time - command_start_time) < 1.0
            ):
                db.outputs.pickandplace_command = True
                db.outputs.prim_name = last_message
            else:
                db.outputs.pickandplace_command = False
                # 새 메시지가 들어오지 않으면 기존 마지막 메시지는 그대로 출력
                if last_message is not None:  # last_message가 None이 아닐 때만 설정
                    db.outputs.prim_name = last_message
                    print(f"DEBUG: Keeping prim_name as {last_message}")
                else:
                    db.outputs.prim_name = "/world/metacombot/plastic_bin"  # 기본값 설정
                    print("DEBUG: Setting default prim_name")
                # 1초 경과 후 타이머는 초기화 (새 메시지 수신시 다시 업데이트됨)
                command_start_time = None

        except Exception as error:
            # db.log_error(str(error))
            return False

        return True
