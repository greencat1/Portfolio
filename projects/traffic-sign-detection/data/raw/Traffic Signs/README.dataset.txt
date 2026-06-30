# Traffic Signs Detection Europe > Well Trained Model
https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-detection-europe

Provided by a Roboflow user
License: CC BY 4.0

## Dataset structure
The dataset contains 55 classes of unique traffic signs. 
There are 4 main categories:
* forb
* info
* mand
* warn

Each class has one of these prefixes in it's name indicating the category of the traffic sign: forbidden, informational, mandatorry and warning, respectively.

Each image is of size 640x640px.
If a traffic sign was recognizable it was labeled but if I could barely tell what the sign was it was not labeled, no matter the distance to the traffic sign.
The dataset is not perfectly balanced due to the frequency of traffic signs. Naturally some signs appear more than others.
There are some images with no labels, those were collected when models would get false positive detections.
## Data collection
Most of the images were collected from Google Maps, Romania, and manually labeled.
There's a small part of images that were automatically collected from YouTube videos, by a trained model, and then manually checked (around 150). 

