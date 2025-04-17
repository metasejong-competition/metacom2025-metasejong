import numpy as np
import omni.graph.core as og
import carb

def quat_mult(q1, q2):
    """
    두 quaternion (형태: [w, x, y, z])의 곱셈 수행 (q_out = q1 * q2)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return q
    return q / norm


class OgnOrientRevert:
    """

    """
    @staticmethod
    def compute(db: og.Database):
        try:
            # 1. 입력 quaternion 읽기 (OmniGraph는 [x, y, z, w] 형식으로 제공)
            q_input = db.inputs.orientation
            x, y, z, w = q_input  # [x, y, z, w] -> 요소 분리

            # 2. [x, y, z, w] → [w, x, y, z]로 변환
            q = np.array([w, x, y, z], dtype=float)
            q = normalize_quat(q)

            # 3. x축 180° 회전을 나타내는 quaternion: [0, 1, 0, 0]
            q_x = np.array([0.0, 1.0, 0.0, 0.0])

            # 4. quaternion 곱셈 적용 (q_x * q)
            q_out = quat_mult(q_x, q)
            q_out = normalize_quat(q_out)

            # 5. [w, x, y, z] → [x, y, z, w]로 변환하여 출력
            w, x, y, z = q_out.tolist()
            db.outputs.reverted_orientation = np.array([x, y, z, w])  # OmniGraph 기대 포맷

        except Exception as e:
            db.log_error("Error in compute: " + str(e))
            return False

        return True
