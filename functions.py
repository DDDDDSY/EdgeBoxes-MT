import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_boxes(boxes, frame):

    frame2 = frame.copy()

    for box in boxes:
        frame2 = cv2.rectangle(frame2, (box[0]-1, box[1]-1), (box[0]+box[2]-1, box[1]+box[3]-1), (255,255,0), 2)

    return frame2
