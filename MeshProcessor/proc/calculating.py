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
    # m_length = m_method(measure=target, opposite=opposite, image=images[m_dir],
    #                     range=m_range, pivot=pivot, depth=m_depth)
    if kwargs['pivot'] < 0:
        return 0.0

    # 여기는 두 개의 resource를 반영하면 안되는 제약 조건이 존재
    # 이게 픽셀 위치
    gap = kwargs['range'][1] - kwargs['range'][0]
    center = int((kwargs['measure'].max() - kwargs['measure'].min()) / 2)
    center_point = center + kwargs['measure'].min()
    right_point = center + int(center * gap) + kwargs['measure'].min()
    left_point = center + -1 * int(center * gap) + kwargs['measure'].min()

    # 세 점의 인덱스(x, y 쌍을 구하기 위함)
    center_index = np.where(kwargs['measure'] == center_point)[0]
    left_index = np.where(kwargs['measure'] == left_point)[0]
    right_index = np.where(kwargs['measure'] == right_point)[0]

    # 세 점의 라인
    width = right_point - left_point

    if 0.0 < kwargs['pivot'] < 1.0:
        line = int((kwargs['opposite'].max() - kwargs['opposite'].min()) * kwargs['pivot'] + kwargs['opposite'].min())
        pos = kwargs['opposite'][line]
        
        # Depth가 없으면 Center쪽으로 좀 더 다가가는 코드가 필요
        # Depth가 애초에 잘못 저장되어 있었음
        height = kwargs['depth'][center_point, pos] - abs((kwargs['depth'][left_point, pos] + kwargs['depth'][right_point, pos]) / 2.0)
    else:
        center_line = kwargs['opposite'][center_index]
        left_line = kwargs['opposite'][left_index]
        right_line = kwargs['opposite'][right_index]
        if kwargs['pivot'] == 0.0:
            height = center_line.max() - abs((left_line.max() - right_line.max()) / 2.0)
        else:
            height = center_line.min() - abs((left_line.min() - right_line.min()) / 2.0)

    a = height
    b = width
    sqrt_term = np.sqrt((b ** 2) + (16 * (a ** 2)))
    front_term = 1 / 2 * sqrt_term
    mid_term = (b ** 2) / (8 * a)
    back_term = np.log((4 * a + sqrt_term) / b)
    return mid_term * back_term + front_term

def get_straight_length(**kwargs):
    if kwargs['pivot'] < 0:
        length = (kwargs['measure'].max() - kwargs['measure'].min()) * (kwargs['range'][1] - kwargs['range'][0])
    else:
        line = int((kwargs['opposite'].max() - kwargs['opposite'].min()) * kwargs['pivot'] + kwargs['opposite'].min())
        pos = kwargs['opposite'][line]
        uq_measure = np.unique(kwargs['measure'])
        image_line = kwargs['image'][uq_measure, pos]
        denoise_line = np.where(image_line != 0)[0]
        filter_line = uq_measure[denoise_line]
        length = (filter_line.max() - filter_line.min()) * (kwargs['range'][1] - kwargs['range'][0])

    return length


def measure_bodies2(**kwargs):
    if kwargs['template'] is None:
        template = YamlConfig.get_dict(r'config/Measurement.yaml')
        template = template['measurement']

    images = copy.deepcopy(kwargs['images'])
    pcds = copy.deepcopy(kwargs['pcds'])
    masks = copy.deepcopy(kwargs['masks'])
    depth = copy.deepcopy(kwargs['depth'])
    # smplx_head = copy.deepcopy(kwargs['smplx_head'])
    resolution = kwargs['res']
    measure_info = template[template['target']]

    output = copy.deepcopy(measure_info)
    for code, value in measure_info.items():
        direction = value['direction']
        resources = value['resource']
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
                # 해당 Part의 정보 가져오기
                surface_height = np.array([], dtype=np.int64)
                surface_width = np.array([], dtype=np.int64)
                for m_resource in resources:
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
                    m_depth = None
                    m_method = get_straight_length

                # 범위 추출
                image = images[m_dir][:, :, 0]
                image = np.array(image * 255, dtype=np.uint8)
                if "w" in method:
                    image = np.rot90(image, 1)
                    if m_depth is not None:
                        m_depth = np.rot90(m_depth, 1)
                    target = surface_width
                    opposite = surface_height
                else:
                    target = surface_height
                    opposite = surface_width

                m_length = m_method(measure=target, opposite=opposite, image=image,
                                    range=m_range, pivot=pivot, depth=m_depth)
                if "2x" in method:
                    m_length = m_length * 2
                length += m_length
        # iamge offset 때문에 2.4cm(0.024) 더 해야함
        meter = length / resolution + 0.024
        output[code] = copy.deepcopy(round(meter * 100, 2))
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
