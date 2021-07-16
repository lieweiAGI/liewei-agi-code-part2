
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 10

ANCHORS_GROUP = {
    13: [[51, 22], [52, 22], [53, 22]],
    26: [[54, 22], [55, 22], [56, 22]],
    52: [[57, 22], [58, 22], [59, 22]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
