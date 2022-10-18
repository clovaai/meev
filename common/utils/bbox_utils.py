

# Convert bbox from coco (left, upper, width, height) to pil (left, upper, right, lower)
def coco_to_pil(bbox):
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])