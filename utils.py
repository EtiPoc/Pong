from collections import namedtuple
import cv2

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


def rgb2gray(rgb):
    rgb = rgb[34:-16]
    rgb = cv2.resize(rgb, (80,80))
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, 80, 80)
