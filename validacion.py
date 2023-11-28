import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def yolo8_to_array(yolo8_filename: str, __last_index=[0]) -> np.array:
    """Array columns are: Index, X, Y, Width, Height, Class"""
    boxes = []
    with open(yolo8_filename, "r") as yolo8_file:
        for i, detection in enumerate(yolo8_file.readlines(), __last_index[0]+1):
            cls, *xywh = detection.split()
            boxes.append(np.fromiter([i] + xywh + [cls], float))
        __last_index[0] = i
    return np.array(boxes)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def show_img(img, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.imshow(img, extent=(0, 1, 0, 1), aspect='auto', **kwargs)
    # ax.set_xticks([])
    # ax.set_yticks([])


def show_img_g(img_g, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.imshow(img_g, cmap='gray', extent=(0, 1, 0, 1), aspect='auto', **kwargs)


def draw_box(x, y, w, h, c, center=True, ax=None, flip_y=True, **kwargs):
    if ax is None:
        ax = plt.axes()

    if flip_y is True:
        y = 1-y

    c = int(c)

    kwargs['ec'] = kwargs.get('color', f"C{c}")  # default color based on class
    kwargs['label'] = kwargs.get('label', None)

    ax.add_patch(
        plt.Rectangle(
            # x, y are centers so this converts to the correct corner (y was saved as top-down but Rectangle need bot-up so y -> 1-y)
                (x-w/2, y-h/2),
            w,
            h,
            fc='none',    # empty box
            **kwargs
        )
    )

    if center is True:
        ax.text(
            x, y, c, c=kwargs['ec'], horizontalalignment='center', verticalalignment='center')

    return ax


def draw_boxes(boxes: np.array, center=True, ax=None, flip_y=True, show=True, **kwargs):
    if ax is None:
        ax = plt.axes()
    for i, *xywhc in boxes:
        label = kwargs.get('label', None)
        if label is not None:
            kwargs.pop('label')
        draw_box(*xywhc, center=center, ax=ax,
                 flip_y=flip_y, label=label, **kwargs)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # may use ax.set_axis_off() to remove black outline box
    ax.set_axis_off()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.tight_layout()
    if show:
        plt.show()


def intersection(boxes1, boxes2):
    """Calculate the intersectional area of two bounding boxes given their centers, widths and heights:

     ┌───────o q1
     │       │
     │    ┌──┼────o  q2
     │    │xx│    │         <- xx = intersection
     o────┼──┘    │
     p1   │       │
          o───────┘
          p2

    """

    # xywh to corner points
    (p1, q1), (p2, q2) = [((x-w/2, y-h/2), (x+w/2, y+h/2))
                          for i, x, y, w, h, c in (boxes1, boxes2)]

    # check if overlap exists
    if all((p1[0] < q2[0], q1[0] > p2[0], p1[1] < q2[1], q1[1] > p2[1])):
        # get middle coords
        _, x1, x2, _ = sorted((p[0] for p in (p1, p2, q1, q2)))
        _, y1, y2, _ = sorted((p[1] for p in (p1, p2, q1, q2)))

        return (x2-x1) * (y2-y1)
    return 0


def iou(boxes1, boxes2):

    a1, a2 = (w*h for i, x, y, w, h, c in (boxes1, boxes2))
    i = intersection(boxes1, boxes2)

    return i / (a1+a2-i)


def plot_centers(real_x, real_y, plan_x, plan_y, title="Centers distribution"):
    g = sns.JointGrid()

    # common (center) scatterplots of box centers
    sns.scatterplot(x=real_x, y=real_y, c='red',
                    ec='none', ax=g.ax_joint, label='real')
    sns.scatterplot(x=plan_x, y=plan_y,  c='blue',
                    ec='none', ax=g.ax_joint, label='plan')

    # x density plots
    sns.kdeplot(x=real_x, ax=g.ax_marg_x, bw_adjust=.2, color='red')
    sns.kdeplot(x=plan_x, ax=g.ax_marg_x, bw_adjust=.2,
                color='blue', fill=True, alpha=0.3)

    # y density plots
    sns.kdeplot(y=real_y, ax=g.ax_marg_y, bw_adjust=.2, color='red')
    sns.kdeplot(y=plan_y, ax=g.ax_marg_y, bw_adjust=.2,
                color='blue', fill=True, alpha=0.3)

    g.fig.suptitle(title)
    g.fig.supxlabel("x")
    g.fig.supylabel("y")
    g.fig.set_tight_layout(True)
    plt.show()

    return g


def transform(real_boxes: np.array, plan_mean=0.0, plan_std=1.0, return_stats=True) -> np.array:
    # unpack
    real_i, real_x, real_y, real_w, real_h, real_c = real_boxes.T

    # calculate extremes of boxes (left/right-ward)
    real_min, real_max = min(real_x - real_w/2), max(real_x + real_w/2)
    real_std, real_mean = real_y.std(), real_y.mean()

    # transform real
    # MinMax both real and plan in x axis
    new_x = (real_x-real_min)/(real_max-real_min)
    new_w = real_w/(real_max-real_min)            # Scale widths accordingly
    new_y = (real_y-real_mean)/real_std           \
        * plan_std + plan_mean                    # Normalize real to match plan's distribution
    new_h = real_h/real_std * plan_std            # Scale heights accordingly

    if return_stats is True:
        return np.array([real_i, new_x, new_y, new_w, new_h, real_c]).T, real_min, real_max, real_mean, real_std
    return np.array([real_i, new_x, new_y, new_w, new_h, real_c]).T


def inverse_transform(transformed_boxes: np.array, x_min, x_max, old_y_mean, old_y_std, new_y_mean, new_y_std) -> np.array:
    # unpack
    i, x, y, w, h, c = transformed_boxes.T

    # inverse transform
    old_x = x * x_max + (1 - x) * x_min
    old_w = w * (x_max - x_min)
    old_y = (y - old_y_mean) / old_y_std * new_y_std + new_y_mean
    old_h = h * new_y_std / old_y_std

    return np.array([i, old_x, old_y, old_w, old_h, c]).T


# -> tuple[np.array, float, float] | np.array:
def preprocess_plan(plan_boxes: np.array, return_stats=True):
    plan_i, plan_x, plan_y, plan_w, plan_h, plan_c = plan_boxes.T
    plan_min, plan_max = min(plan_x - plan_w/2), max(plan_x + plan_w/2)

    new_x = (plan_x - plan_min) / (plan_max - plan_min)  # MinMax x axis
    new_w = plan_w / (plan_max-plan_min)                # rescale widths

    new_plan = np.array([plan_i, new_x, plan_y, new_w, plan_h, plan_c]).T
    if return_stats:
        return new_plan, plan_y.mean(), plan_y.std()
    return new_plan


def box_distance(box1, box2):
    _, x1, y1, *_ = box1
    _, x2, y2, *_ = box2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def match_boxes(real_boxes: np.array, plan_boxes: np.array, default_class=0, overlap_threshold=0.05, distance_threshold=.2) -> np.array:
    # real_box: plan_box
    mapping, swap, missing = {}, {}, []
    real_available = np.full_like(real_boxes[:, 0], True, dtype=bool)
    # for each box in the plan
    for plan_box in plan_boxes:
        # get the index and class
        i, *_, cls = plan_box
        # mask same class
        class_mask = real_boxes[:, -1] == cls
        # candidates are real available boxes from the correct class
        candidates = real_boxes[real_available & class_mask]
        # calculate the iou against all candidates
        ious = np.fromiter(
            (iou(plan_box, candidate)
                for candidate in candidates),
            float
        )
        if ious.any():
            # choose the highest one
            best_idx = ious.argmax()
            # update mapping and available real boxes
            if ious[best_idx] > overlap_threshold:
                mapping[candidates[best_idx, 0]] = i
                real_available &= real_boxes[:, 0] != candidates[best_idx, 0]
                continue
        # if no overlap was higher than 'overlap_threshold' then repeat to get the closest box
        distances = np.fromiter(
            (box_distance(plan_box, candidate)
                for candidate in candidates),
            float
        )
        if distances.any():
            close_idx = distances.argmax()
            if distances[close_idx] < distance_threshold:
                mapping[candidates[close_idx, 0]] = i
                real_available &= real_boxes[:, 0] != candidates[close_idx, 0]
                continue

        # if no intersection was big enough and no distance small enough the product is missing
        # try to figure out what is in its place (a real unassigned box)
        all_real_boxes = real_boxes[real_available]
        all_ious = np.fromiter(
            (iou(plan_box, box)
                for box in all_real_boxes),
            float
        )
        if all_ious.any():
            # choose the highest one
            swap_idx = all_ious.argmax()
            # update swap and available real boxes
            if all_ious[swap_idx] > overlap_threshold:
                swap[all_real_boxes[swap_idx, 0]] = i
                real_available &= real_boxes[:,
                                             0] != all_real_boxes[swap_idx, 0]
                continue

        missing.append(i)

    extra = real_boxes[real_available, 0]

    return mapping, swap, missing, extra


def plot_matches(real_boxes: np.array, plan_boxes: np.array, matches: dict, swap: dict, missing: list, extra: list, real_img: np.array = None, default_class=0, save_dir=None, flip_y=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    if real_img is not None:
        show_img(real_img, ax=ax)

    # TODO: join 'filtered_boxes', 'plot_kwargs' and 'draw_boxes' in a same loop
    filtered_boxes = map(lambda indices, boxes: boxes[np.in1d(boxes[:, 0], indices), :],
                         (list(swap.keys()), missing, extra),       # list(matches.keys()), 
                         (real_boxes, plan_boxes, real_boxes))      # real_boxes

    plot_kwargs = (
        #dict(color='limegreen', label='Match'),
        dict(color='blueviolet', label='Swapped'),
        dict(color='red', label='Missing'),
        dict(color='mediumvioletred', label='Extra')
    )

    for boxes, kwargs in zip(filtered_boxes, plot_kwargs):
        draw_boxes(boxes, ax=ax, flip_y=flip_y, center=False,
                   show=False, **kwargs, linewidth=2)

    # ax.legend(loc='center left', bbox_to_anchor=(
    #     0, -0.05, 1, -.05), mode='expand', ncols=4)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(save_dir + '/output.png')
        return '/output.png'


def calculate_score(matches, swapped, missing, extra):
    matches, swapped, missing, extra = map(
        len, (matches, swapped, missing, extra))
    return matches / (matches + swapped + missing + extra)


def get_pos(box):
    i, x, y, w, h, c = box.T
    if x < 1/3 and y < 1/3:
        return "Abajo a la izquierda"
    if 1/3 < x < 2/3 and y < 1/3:
        return "Abajo en el centro"
    if x > 2/3 and y < 1/3:
        return "Abajo a la derecha"

    if x < 1/3 and 1/3 < y < 2/3:
        return "En el centro a la izquierda"
    if 1/3 < x < 2/3 and 1/3 < y < 2/3:
        return "En el centro"
    if x > 2/3 and 1/3 < y < 2/3:
        return "En el centro a la derecha"

    if x < 1/3 and y > 2/3:
        return "Arriba a la izquierda"
    if 1/3 < x < 2/3 and y > 2/3:
        return "Arriba en el centro"
    if x > 2/3 and y > 2/3:
        return "Arriba a la derecha"


def format_error_mesages(swapped, missing, extra, plan_boxes, real_boxes, product_dict):
    fmts = {
        "swapped": "{pos} se detectó {prod1} en lugar de {prod2}",
        "missing": "{pos} se detectó {prod} faltante",
        "extra": "{pos} se detectó {prod} sobrante"
    }
    
    errors = []

    for real_id, plan_id in swapped.items():
        errors.append(
            fmts['swapped'].format(
                pos= get_pos(real_boxes[real_boxes[:, 0] == real_id]),
                prod1= product_dict[real_boxes[real_boxes[:, 0] == real_id, -1][0]],
                prod2= product_dict[plan_boxes[plan_boxes[:, 0] == plan_id, -1][0]]
            )
        )
    
    for real_id in extra:
        errors.append(
            fmts['extra'].format(
                pos= get_pos(real_boxes[real_boxes[:, 0] == real_id]),
                prod= product_dict[real_boxes[real_boxes[:, 0] == real_id, -1][0]]
            )
        )

    for plan_id in missing:
        errors.append(
            fmts['missing'].format(
                pos= get_pos(plan_boxes[plan_boxes[:, 0] == plan_id]),
                prod= product_dict[plan_boxes[plan_boxes[:, 0] == plan_id, -1][0]]
            )
        )

    return errors
    

def test_pipeline(plan_boxes, real_boxes, product_dict, real_img=None, save_dir=None, overlap_threshold=.25, distance_threshold=.1):

    # transform the plan and extract stats
    new_plan_boxes, plan_mean, plan_std = preprocess_plan(plan_boxes)
    
    # transform the real using plan's stats
    new_real_boxes, real_x_min, real_x_max, real_y_mean, real_y_std = transform(
        real_boxes, plan_mean, plan_std, return_stats=True)
    
    # match transformations
    matches, swapped, missing, extra = match_boxes(
        new_real_boxes, new_plan_boxes, overlap_threshold=overlap_threshold, distance_threshold=distance_threshold)
    
    # inverse transform plan boxes to better match realogram
    plot_plan_boxes = inverse_transform(
        new_plan_boxes, real_x_min, real_x_max, plan_mean, plan_std, real_y_mean, real_y_std)

    # save img
    processed_img_path = plot_matches(
        real_boxes, plot_plan_boxes, matches, swapped, missing, extra, real_img, save_dir=save_dir)

    score = calculate_score(matches, swapped, missing, extra)
    errors = format_error_mesages(
        swapped, missing, extra, plan_boxes, real_boxes, product_dict)

    return processed_img_path, score, errors