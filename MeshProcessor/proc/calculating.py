import copy
import numpy as np
from util.yaml_config import YamlConfig


def check_float(element):
    partition = element.partition('.')

    if element.isdigit():
        return True

    elif (partition[0].isdigit() and partition[1] == '.' and partition[2].isdigit()) or (
            partition[0] == '' and partition[1] == '.' and partition[2].isdigit()) or (
            partition[0].isdigit() and partition[1] == '.' and partition[2] == ''):
        return True
    else:
        return False


def get_theta_from(a, b):
    y_dist = b[1] - a[1]
    z_dist = b[2] - a[2]
    return np.arctan(z_dist / y_dist)


def get_parabola_length(**kwargs):
    a = kwargs['height']
    b = kwargs['length']
    sqrt_term = np.sqrt((b ** 2) + (16 * (a ** 2)))
    front_term = 1 / 2 * sqrt_term
    mid_term = (b ** 2) / (8 * a)
    back_term = np.log((4 * a + sqrt_term) / b)
    return mid_term * back_term + front_term


def get_straight_length(**kwargs):
    temp_array = a - b
    euc = np.linalg.norm(temp_array)
    return euc


def measure_bodies2(**kwargs):
    if kwargs['template'] is None:
        template = YamlConfig.get_dict(r'config/Measurement.yaml')
        template = template['measurement']

    images = copy.deepcopy(kwargs['images'])
    pcds = copy.deepcopy(kwargs['pcds'])
    masks = copy.deepcopy(kwargs['masks'])
    depth = copy.deepcopy(kwargs['depth'])
    smplx_head = copy.deepcopy(kwargs['smplx_head'])
    resolution = kwargs['res']
    check_points = template['check_points']
    measure_info = template[template['target']]

    output = copy.deepcopy(measure_info)
    for code, value in check_points.items():
        direction = value['direction']
        resources = value['resources']
        method = value['method']
        m_range = value['range']
        pivot = value['pivot']
        length = 0
        if "smplx" in direction:
            output[code] = 0.0
        else:
            # load data
            for m_dir in direction.split("|"):
                mask = masks[m_dir][:, :, 1]
                for m_resource in resources:
                    # 해당 Part의 정보 가져오기
                    surface_height = np.array([], dtype=np.int64)
                    surface_width = np.array([], dtype=np.int64)

                    for color, part_label in zip(template['custom_colors'], template['part_labels']):
                        if m_resource not in part_label.lower():
                            continue
                        part_x, part_y = np.where(mask == color[1])
                        surface_height = np.concatenate([surface_height, part_x])
                        surface_width = np.concatenate([surface_width, part_y])

                # 알고리즘 정의
                if "outline" in method:
                    m_depth = depth[m_dir]
                    m_method = get_parabola_length
                else:
                    m_method = get_straight_length

                # 범위 추출
                if "w" in method:
                    target = surface_width
                    opposite = surface_height
                else:
                    target = surface_height
                    opposite = surface_width
                line = int(opposite.shape[0] * pivot)
                pos = opposite[line] # 반대편을 통해 값을 쭉 빼올 수 있다
                target # 얘로 길이를 잴거다
                # 그러니 가운데 뎁스를 알 수 있다
                # 이미지로 해야 위치잡기가 쉽다
                #
                m_length = 0
                if "2x" in method:
                    m_length = m_length * 2
                length += m_length
        output[code] = length
    return output

def measure_bodies(**kwargs):
    if kwargs['template'] is None:
        template = YamlConfig.get_dict(r'config/Measurement.yaml')
        template = template['measurement']

    images = copy.deepcopy(kwargs['images'])
    pcds = copy.deepcopy(kwargs['pcds'])
    masks = copy.deepcopy(kwargs['masks'])
    depth = copy.deepcopy(kwargs['depth'])
    resolution = kwargs['res']

    check_points = template['check_points']
    rely_on = template['rely_on']
    output = copy.deepcopy(rely_on)
    for key, value in output.items():
        output[key] = 0.0

    for direction_length, parts in check_points.items():
        sep = direction_length.split('_')
        direction = sep[0]

        # 대상 direction 추출
        mask = masks[direction][:, :, 1]
        z_set = depth[direction]
        image = images[direction]

        length = sep[1]
        for part, thresholds in parts.items():
            # 해당 Part의 정보 가져오기
            surface_height = np.array([], dtype=np.int64)
            surface_width = np.array([], dtype=np.int64)
            for color, part_label in zip(template['custom_colors'], template['part_labels']):
                if part not in part_label.lower():
                    continue
                # 부위 구하기
                part_x, part_y = np.where(mask == color[1])
                surface_height = np.concatenate([surface_height, part_x])
                surface_width = np.concatenate([surface_width, part_y])

            real_x, real_y, real_z, rgb = [], [], [], []
            for x, y in zip(surface_height, surface_width):
                if image[x, y, 0] != 0.0:
                    real_x.append(x)
                    real_y.append(y)
                    real_z.append(copy.deepcopy(z_set[x, y]))
                    rgb.append(np.array(image[x, y, :] * 255, dtype=np.uint8))

            real_x, real_y, real_z = np.array(real_x), np.array(real_y), np.array(real_z)
            real_x = mask.shape[0] - real_x
            real_x = real_x / resolution
            real_x = real_x - real_x.min()

            real_y = real_y / resolution
            real_y = real_y - real_y.min()

            real_z = real_z - real_z.min()

            # 길이, 높이 재기(단 pixel의 경우, )
            for threshold, code in thresholds.items():
                if check_float(threshold):
                    threshold = float(threshold)
                    pivot = max(real_x) * threshold
                    if length == 'width':
                        cond1 = np.where(real_x > (pivot - 0.01))
                        cond2 = np.where(real_x > (pivot + 0.01))
                        cond = np.intersect1d(cond1, cond2)
                        target = real_y[cond]
                        value = max(target) - min(target)
                    else:
                        value = pivot
                    output[code] = copy.deepcopy(round(value.item() * 100, 2))
                else:
                    value = template['reserved'][threshold]
                    output[code] = copy.deepcopy(round(value * 100, 2))

    output["X_Measure_1000"] = copy.deepcopy(kwargs['total_height'])
    YamlConfig.write_yaml('./Measure.yaml', output)
    return output
