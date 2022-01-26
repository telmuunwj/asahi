import requests
import os

url = "https://onedrive.live.com/download?cid=74086755007FD274&resid=74086755007FD274%21108&authkey=ALUKJ5p70bSfKwk"
dest = os.getcwd() + "/models/mask_rcnn_coco_0032.h5"
print("Downloading MaskRCNN model...")
r = requests.get(url)
open(dest , 'wb').write(r.content)
