from moviepy.editor import VideoFileClip
import classLine
#from processFrame import process_frame
#from processFrame_colorgrad import process_frame
import datetime as dt
import time
import cv2


# Create instance of class
frame = classLine.Line()


#start_time = dt.datetime.now()

#video_in = VideoFileClip("../../ac_laguna_mx5_2.mp4")
#clip = video_in.fl_image(frame.process_frame)
#clip = video_in.fl_image(frame.process_frame).subclip(15,25)
#clip.write_videofile('../output_videos/output_video.mp4', audio=False)
#cv2.imshow('TEST_WINDOW', output)

cap = cv2.VideoCapture("../../ac_laguna_mx5_2.mp4")

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        cv2.imshow('Output', frame.process_frame(img))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

#end_time = dt.datetime.now()
#elapsed_time = end_time - start_time
#print("Elapsed Time: ", elapsed_time)

'''
start_time = dt.datetime.now()

video_in = VideoFileClip("../challenge_P1.mp4")
clip = video_in.fl_image(frame.process_frame)
clip.write_videofile('../output_videos/output_challenge_P1.mp4', audio=False)

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print("Elapsed Time: ", elapsed_time)


start_time = dt.datetime.now()

video_in = VideoFileClip("../challenge_video.mp4")
clip = video_in.fl_image(frame.process_frame)
clip.write_videofile('../output_videos/output_challenge_video.mp4', audio=False)

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print("Elapsed Time: ", elapsed_time)


start_time = dt.datetime.now()

video_in = VideoFileClip("../harder_challenge_video.mp4")
clip = video_in.fl_image(frame.process_frame)
clip.write_videofile('../output_videos/output_harder_challenge_video.mp4', audio=False)

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print("Elapsed Time: ", elapsed_time)
'''







"""
#time.sleep(3)

end_time = dt.datetime.now()

elapsed_time = end_time - start_time

#print("Elapsed Time: ", (elapsed_time.microseconds)/1e6)
print("Elapsed Time: ", elapsed_time)
"""
