# Social Distance Detection

## Objective: Detect whether the pedestrians are maintaining social distance or not. Also give the number of violence.

## Project Pipeline :
1. Get video input from webcam or video file.
2. Use a pre-trained yolov3 model to detect pedestrians.
3. Measure the centroid of pedestrians using the bounding box co-ordinate, which is taken from the output of yolov3 model.
4. Then map the pixel value to distance.
5. Measure the distance between all centroid points.
6. If the distance is less than the threshold value, pedestrians action is marked as violence.
7. Show the output with co-responding bounding boxes and number of violence.

## To see the output of the project, please click [here](https://youtu.be/rbtjcTKr-cI)