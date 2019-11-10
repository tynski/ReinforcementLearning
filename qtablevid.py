import cv2


def make_video():
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(0, 4000, 100):
        img_path = "qtable_charts/{}.png".format(i)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video()