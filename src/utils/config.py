# Image size for the camera
# > 224 for mobilenet and resnet
IMG_SIZE = 224

# Frames per second for video reading
FPS_VIDEO_READING = 30

# Number of top predictions to return
TOP_K_PREDICTIONS = 2

# Mean of image channels for normalization
# > [0.485, 0.456, 0.406] values for mobilenet and resnet
mean_ch_R = 0.485
mean_ch_G = 0.456
mean_ch_B = 0.406
mean_channels = [mean_ch_R, mean_ch_G, mean_ch_B]

# Standard deviation of image channels for normalization
# > [0.229, 0.224, 0.225] values for mobilenet and resnet
std_ch_R = 0.229
std_ch_G = 0.224
std_ch_B = 0.225
std_channels = [std_ch_R, std_ch_G, std_ch_B]
