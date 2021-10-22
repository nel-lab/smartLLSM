#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:51:24 2021

@author: jimmytabet
"""

#%% imports
from shapely.geometry.point import Point
from shapely import affinity
import numpy as np

#%% create ellipse function
def create_ellipse(cx,cy,a,b,theata):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circle = Point((cx,cy)).buffer(1)
    ellipse = affinity.scale(circle, a, b)
    rotated_ellipse = affinity.rotate(ellipse, theata)
    return rotated_ellipse

#%% create ellipses
cx1=400
cy1=400
a1=20
b1=40
theta1=10
A1 = np.pi*a1*b1
ellipse1 = create_ellipse(cx1,cy1,a1,b1,theta1)

cx2=400
cy2=400
a2=30
b2=20
theta2=50
A2 = np.pi*a2*b2
ellipse2 = create_ellipse(cx2,cy2,a2,b2,theta2)

intersection = ellipse1.intersection(ellipse2).area
union = ellipse1.union(ellipse2).area
# potentially faster
# union = A1+A2-intersection

print('area of ellipse 1:', ellipse1.area, A1)
print('area of ellipse 2:', ellipse2.area, A2)
print('area of intersect:', intersection)
print('area of union:', union, A1+A2-intersection)

#%% 
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = np.zeros((800,800)) # convert to FOV resolution (800x800)

cx1=400
cy1=400
a1=20
b1=40
theta1=10
A1 = np.pi*a1*b1
ellipse1 = cv2.ellipse(np.zeros((800,800)), (cx1,cy1), (a1,b1) ,theta1, 0, 360, 1, -1)

cx2=400
cy2=400
a2=30
b2=20
theta2=50
A2 = np.pi*a2*b2
ellipse2 = cv2.ellipse(np.zeros((800,800)), (cx2,cy2), (a2,b2) ,theta2, 0, 360, 1, -1)

plt.subplot(131)
plt.imshow(ellipse1)
plt.title(f'sum: {np.sum(ellipse1)}, calc: {np.round(A1,0)}')

plt.subplot(132)
plt.imshow(ellipse2)
plt.title(f'sum: {np.sum(ellipse2)}, calc: {np.round(A2,0)}')

combined = ellipse1 + ellipse2
intersection = np.sum(combined == 2)
union = np.sum(combined > 0)

plt.subplot(133)
plt.imshow(combined)
plt.title(intersection/union)

print(f'IoU = {intersection/union}')