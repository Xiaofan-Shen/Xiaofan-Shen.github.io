---
layout: post
title:  "Face morping"
date:   2022-04-07 00:00:00 +0000
tags: Robotics VR Haptic
color: rgb(200,200,200)
cover: '/assets/photos/face.png'
# subtitle: ''
---
This project 'face morphing' is divided into 7 parts to complete a smooth transfer video from two
faces;
1. 'dlib' face landmark detector is used to detect 68 landmarks on 2 images, as 68 landmarks are
not covering all the face, 20 landmarks are manually added, including 16 landmarks on the face,
while 4 points locates the background, so that the background could also transfer smoothly;
2. the standard Delaunay Triangulation is used to visualize triangulation of vertices(landmarks);
3. get the interpolating between image1 and image2 by applying
x int = (x img1 + ximg2)/2
y int = (y img1 + yimg2)/2
4. Getting parameters of the affine warp when vertices of two triangle were given, and between
every pairs of corresponding triangles, Affine warp was estimated;
5. Find the all coordinates which are inside the given triangle, and then a labeled image was created
by dividing face into parts;
6. Get the in - between image for a given 'w'
7. Create the transfer video;

<iframe type="text/html" width="100%" height="385" src="https://www.youtube.com/embed/0kpqJpmSTOg" frameborder="0"></iframe>
