## get a list for literal annotations of given image
from a3ds import A3DS
# literal description from Polina Tsvilodub
dict_factor_rules = {
    "floor_hue": {
    0: ["FLOOR_HUE -> 'red'"], # more synonyms / rules can be added here
    0.1: ["FLOOR_HUE -> 'orange'"],
    0.2: ["FLOOR_HUE -> 'yellow'"],
    0.30000000000000004: ["FLOOR_HUE -> 'green'"],
    0.4: ["FLOOR_HUE -> 'light green'"],
    0.5: ["FLOOR_HUE -> 'cyan'"],
    0.6000000000000001: ["FLOOR_HUE -> 'medium blue'"],
    0.7000000000000001: ["FLOOR_HUE -> 'dark blue'"],
    0.8: ["FLOOR_HUE -> 'purple'"],
    0.9: ["FLOOR_HUE -> 'pink'"]
    },
    "wall_hue": {
    0: ["WALL_HUE -> 'red'"], # more synonyms / rules can be added here
    0.1: ["WALL_HUE -> 'orange'"],
    0.2: ["WALL_HUE -> 'yellow'"],
    0.30000000000000004: ["WALL_HUE -> 'green'"],
    0.4: ["WALL_HUE -> 'light green'"],
    0.5: ["WALL_HUE -> 'cyan'"],
    0.6000000000000001: ["WALL_HUE -> 'medium blue'"],
    0.7000000000000001: ["WALL_HUE -> 'dark blue'"],
    0.8: ["WALL_HUE -> 'purple'"],
    0.9: ["WALL_HUE -> 'pink'"]
    },
    "object_hue": {
        0: ["OBJECT_HUE -> 'red'"],  # more synonyms / rules can be added here
        0.1: ["OBJECT_HUE -> 'orange'"],
        0.2: ["OBJECT_HUE -> 'yellow'"],
        0.30000000000000004: ["OBJECT_HUE -> 'green'"],
        0.4: ["OBJECT_HUE -> 'light green'"],
        0.5: ["OBJECT_HUE -> 'cyan'"],
        0.6000000000000001: ["OBJECT_HUE -> 'medium blue'"],
        0.7000000000000001: ["OBJECT_HUE -> 'dark blue'"],
        0.8: ["OBJECT_HUE -> 'purple'"],
        0.9: ["OBJECT_HUE -> 'pink'"]
    },

    "scale": {
        0.75: ["SCALE -> 'tiny'"],
        0.8214285714285714: ["SCALE -> 'small'"],
        0.8928571428571428: ["SCALE -> 'medium-sized'"],
        0.9642857142857143: ["SCALE -> 'middle-sized'"],
        1.0357142857142856: ["SCALE -> 'big'"],
        1.1071428571428572: ["SCALE -> 'large'"],
        1.1785714285714286: ["SCALE -> 'huge'"],
        1.25: ["SCALE -> 'giant'"]
    },
    "shape": {  # 0=cube, 1=cylinder, 2=sphere , 3=pill
        0: ["SHAPE -> 'block'"],
        1: ["SHAPE -> 'cylinder'"],
        2: ["SHAPE -> 'ball'"],
        3: ["SHAPE -> 'pill'"]
    },
    "orientation": {  # TODO find better descriptions
        -30: ["ORIENTATION -> 'in the right corner'"],
        -25.714285714285715: ["ORIENTATION -> 'on the right'"],
        -21.42857142857143: ["ORIENTATION -> 'close to the right side'"],
        -17.142857142857142: ["ORIENTATION -> 'near the right corner'"],
        -12.857142857142858: ["ORIENTATION -> 'close to the middle'"],
        -8.571428571428573: ["ORIENTATION -> 'nearly in the middle'"],
        -4.285714285714285: ["ORIENTATION -> 'in the middle'"],
        0: ["ORIENTATION -> 'in the middle'"],
        4.285714285714285: ["ORIENTATION -> 'in the middle'"],
        8.57142857142857: ["ORIENTATION -> 'nearly in the middle'"],
12.857142857142854: ["ORIENTATION -> 'close to the middle'"],
        17.14285714285714: ["ORIENTATION -> 'near the left corner'"],
        21.42857142857143: ["ORIENTATION -> 'close to the left side'"],
        25.714285714285715: ["ORIENTATION -> 'on the left'"],
        30: ["ORIENTATION -> 'in the left corner'"],
    }
}
def get_literal_annotations(item_ID):
    literal_annotations = []
    _,_,_,numeric_lbl,_ = A3DS(item_ID)
    for n in numeric_lbl:
        literal_annotations.append(dict_factor_rules[n.index()][n])

    return literal_annotations




