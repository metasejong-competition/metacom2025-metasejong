"""
This is the implementation of the OGN node defined in OgnDeterminePosition.ogn
"""

import omni.graph.core as og
import random
import numpy as np

# 전역 딕셔너리: 각 아이템은 'type'과 1~10 사이의 랜덤 점수를 가짐.
item_scores = {
    "tissue": {"type": "paper"},
    "juice": {"type": "plastic"},
    "disposable_cup": {"type": "plastic"},
    "wood_block": {"type": "none"},
    "mug": {"type": "none"},
    "cracker_box": {"type": "paper"},
    "cola_can": {"type": "aluminum"},
    "master_chef_can": {"type": "aluminum"},
}


class OgnDeterminePosition:
    @staticmethod
    def compute(db: og.Database):
        # 예: "/world/trash/Chungmu2_002_master_chef_can"
        full_path = db.inputs.target_prim_path

        # 1) 전체 경로에서 마지막 세그먼트 추출
        last_segment = full_path.split("/")[-1] if full_path else ""

        # 2) 언더바('_') 기준으로 분리
        split_parts = last_segment.split("_")

        # 3) 아이템 이름 결정 로직 개선:
        #    마지막 3개, 2개, 1개 조합 순서로 사전에 존재하는 후보를 찾음.
        item_name = None
        for n in range(min(3, len(split_parts)), 0, -1):
            candidate = "_".join(split_parts[-n:])
            if candidate in item_scores:
                item_name = candidate
                break
        if item_name is None:
            # 사전에 없는 경우, 마지막 요소 사용 (또는 기본값 설정)
            item_name = split_parts[-1] if split_parts else ""

        # 기본 오프셋 설정
        placement_pos = np.array([0.6, 0.0, 0])
        pick_offset = np.array([0.0, 0.0, 0])

        # 아이템이 사전에 있으면 위치 오프셋 적용
        if item_name in item_scores:
            item_type = item_scores[item_name]["type"]
            # placement_pos: paper / plastic / aluminum 등으로 구분
            if item_type == "paper":
                placement_pos = np.array([0.6, 0.15, 0])
            elif item_type == "plastic":
                placement_pos = np.array([0.6, 0.0, 0])
            elif item_type == "aluminum":
                placement_pos = np.array([0.6, -0.15, 0])
            else:
                placement_pos = np.array([0.6, 0.0, 0])

            # pick_offset: 아이템 이름에 따라 세부 설정
            if item_name == "juice":
                pick_offset = np.array([0.0, 0.0, -0.02])
            elif item_name in ["cola_can", "master_chef_can"]:
                pick_offset = np.array([0.0, 0.0, 0])
            elif item_name == "tissue":
                pick_offset = np.array([0.0, 0.0, -0.02])
            elif item_name == "disposable_cup":
                pick_offset = np.array([0.0, 0.0, -0.04])
            elif item_name == "wood_block":
                pick_offset = np.array([0.0, 0.0, 0.00])
            elif item_name == "mug":
                pick_offset = np.array([0.0, 0.0, 0])
            elif item_name == "cracker_box":
                pick_offset = np.array([0.0, 0.0, 0.05])
            else:
                pick_offset = np.array([0.0, 0.0, 0])
        else:
            # 사전에 없는 아이템은 기본값
            placement_pos = np.array([0.3, 0.0, 0])
            pick_offset = np.array([0.0, 0.0, 0])

        db.outputs.pick_offset = pick_offset
        db.outputs.placement_offset = placement_pos
        return True
