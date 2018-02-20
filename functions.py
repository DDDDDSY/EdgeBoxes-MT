import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_boxes(boxes, frame):

    for box in boxes:
        frame2 = cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255,0), 3)

    return frame2
