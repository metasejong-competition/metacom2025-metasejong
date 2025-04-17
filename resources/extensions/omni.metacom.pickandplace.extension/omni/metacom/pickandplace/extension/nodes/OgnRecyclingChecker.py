import omni.graph.core as og
import random
import omni.usd

SCORE = 1.0  # 점수 기본값

# 전역 딕셔너리: 각 아이템은 'type'과 1~10 사이의 랜덤 점수를 가짐.
item_scores = {
    "tissue": {"type": "paper", "score": SCORE},
    "juice": {"type": "plastic", "score": SCORE},
    "disposable_cup": {"type": "plastic", "score": SCORE},
    "wood_block": {"type": "none", "score": SCORE},
    "mug": {"type": "none", "score": SCORE},
    "cracker_box": {"type": "paper", "score": SCORE},
    "cola_can": {"type": "aluminum", "score": SCORE},
    "master_chef_can": {"type": "aluminum", "score": SCORE},
}

# 원래 모든 물체 이름을 담은 리스트 (재시작 시 이 값으로 복원됨)
object_list_original = list(item_scores.keys())
object_list = object_list_original.copy()


class OgnRecyclingChecker:
    @staticmethod
    def setup(db: og.Database) -> bool:
        # 노드가 리셋될 때마다 객체 리스트를 원상복구
        global object_list
        object_list = object_list_original.copy()
        return True

    @staticmethod
    def compute(db: og.Database) -> bool:
        try:
            db.outputs.execOut = og.ExecutionAttributeState.ENABLED
            trigger_body = db.inputs.trigger_body
            other_body = db.inputs.other_body

            trigger_last = trigger_body.split("/")[-1] if trigger_body else ""
            other_last = other_body.split("/")[-1] if other_body else ""

            # trigger_material 결정
            if trigger_last.startswith("trigger_"):
                trigger_material = trigger_last[len("trigger_"):]
            else:
                db.log_warning("trigger_body가 예상하는 'trigger_' 접두사를 포함하지 않음. 전체 문자열을 material로 사용합니다.")
                trigger_material = trigger_last

            # 기본값 초기화
            db.outputs.recycling_points = 0.0
            db.outputs.deactivate_prim_path = ""

            # 예) ["Gunja2", "003", "extra", "master", "chef", "can"] (길이 6)
            split_parts = other_last.split("_")
            matched_key = None

            if len(split_parts) >= 4:
                # 마지막 4개 합쳐보기: "extra_master_chef_can"
                candidate4 = "_".join(split_parts[-4:])
                if candidate4 in item_scores:
                    matched_key = candidate4
                else:
                    # 마지막 3개
                    candidate3 = "_".join(split_parts[-3:])
                    if candidate3 in item_scores:
                        matched_key = candidate3
                    else:
                        # 마지막 2개
                        candidate2 = "_".join(split_parts[-2:])
                        if candidate2 in item_scores:
                            matched_key = candidate2
                        else:
                            # 마지막 1개
                            candidate1 = split_parts[-1]
                            if candidate1 in item_scores:
                                matched_key = candidate1

            # 만약 여기까지 못 찾았다면, 기존 방식으로 경로 전체에서 검색
            if matched_key is None:
                for key in item_scores.keys():
                    if key in other_body:
                        matched_key = key
                        break

            # 아이템 매칭
            if matched_key:
                item_info = item_scores[matched_key]
                if item_info["type"] == trigger_material:
                    db.outputs.recycling_points = item_info["score"]
                else:
                    db.outputs.recycling_points = 0.0

                # trash 경로
                if "trash" in other_body:
                    db.outputs.deactivate_prim_path = other_body
                else:
                    db.outputs.deactivate_prim_path = ""

                global object_list
                if matched_key in object_list:
                    object_list.remove(matched_key)
            else:
                db.log_warning("other_body에서 유효한 아이템 키를 찾지 못함")
                db.outputs.recycling_points = 0.0
                db.outputs.deactivate_prim_path = ""

            stage = omni.usd.get_context().get_stage()
            trash = stage.GetPrimAtPath("/world/trash")
            direct_children = trash.GetChildren()
            filtered_count = 0
            for prim in direct_children:
                name = prim.GetName()
                for key, info in item_scores.items():
                    if key in name and info["type"] != "none":
                        filtered_count += 1
                        break
            db.outputs.remained_trashes = filtered_count - 1

        except Exception as error:
            db.log_error(str(error))
            return False

        return True
