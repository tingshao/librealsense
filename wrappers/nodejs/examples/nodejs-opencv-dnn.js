#!/usr/bin/env node

// Copyright (c) 2017 Intel Corporation. All rights reserved.
// Use of this source code is governed by an Apache 2.0 license
// that can be found in the LICENSE file.

/*
const cv = require('../');

const image = cv.imread('../data/got.jpg');
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

// detect faces
const { objects, numDetections } = classifier.detectMultiScale(image.bgrToGray());

if (!objects.length) {
  throw new Error('No faces detected!');
}

// draw detection
const numDetectionsTh = 10;
objects.forEach((rect, i) => {
  const color = new cv.Vec(255, 0, 0);
  let thickness = 2;
  if (numDetections[i] < numDetectionsTh) {
    thickness = 1;
  }

  image.drawRectangle(
    new cv.Point(rect.x, rect.y),
    new cv.Point(rect.x + rect.width, rect.y + rect.height),
    color,
    { thickness }
  );
});

cv.imshowWait('face detection', image);

*/

'use strict';

const rs2 = require('../index.js');
const cv = require('opencv4nodejs');


const {GLFWWindow} = require('./glfw-window.js');
const {glfw} = require('./glfw-window.js');

const inWidth = 300;
const inHeight = 300;
const WHRatio = inWidth / inHeight;
const inScaleFactor = 0.007843;
const meanVal = 127.5;
const classNames = ['background',
                             'aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor'];

function frameToMat(frame) {
  const w = frame.width;
  const h = frame.height;

  if (frame.format === rs2.format.FORMAT_BGR8) {
    return new cv.Mat(h, w, cv.CV_8UC3, frame.data);
  } else if (frame.format === rs2.format.FORMAT_RGB8) {
    let r = new cv.Mat(frame.data, h, w, cv.CV_8UC3);
    r.cvtColor(cv.COLOR_BGR2RGB);
    return r;
  } else if (frame.format === rs2.format.FORMAT_Z16) {
    return new cv.Mat(h, w, cv.CV_16UC1, frame.data);
  } else if (frame.format === rs2.format.FORMAT_Y8) {
    return new cv.Mat(h, w, cv.CV_8UC1, frame.data);
  }
}

const win = new GLFWWindow(1280, 720, 'Node.js Capture Example');

let net = cv.readNetFromCaffe("MobileNetSSD_deploy.prototxt", 
                               "MobileNetSSD_deploy.caffemodel");

const pipeline = new rs2.Pipeline();

// Start the camera
let pipelineProfile = pipeline.start();
let profiles = pipelineProfile.getStreams();
let profile;
profiles.forEach((p) => {
  if (p.streamType === rs2.stream.STREAM_COLOR) {
    profile = p;
  }
})
let align = new rs2.Align(rs2.stream.STREAM_COLOR);
let counter = 0;
while (! win.shouldWindowClose()) {
  const frameset = pipeline.waitForFrames();
  let data = align.process(frameset);
  counter ++;
  if (counter % 10 !== 0) {
    continue;
  }
  let color = data.colorFrame;
  let depth = data.depthFrame;

  let colorMat = frameToMat(color);
  let inputBlob = cv.blobFromImage(colorMat, inScaleFactor,
                                new cv.Size(inWidth, inHeight), new cv.Vec(meanVal, 0, 0), false); //Convert Mat to batch of images

  net.setInput(inputBlob, 'data'); //set the network input
  let detection = net.forward('detection_out'); //compute output
  let detectionMat = detection.flattenFloat(detection.sizes[2], detection.sizes[3]);

  let confidenceThreshold = 0.8;
  for(let i = 0; i < detectionMat.rows; i++) {
      let confidence = detectionMat.at(i, 2);

      if(confidence > confidenceThreshold) {
          let objectClass = detectionMat.at(i, 1);

          console.log(classNames[objectClass]);

          // int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
          // int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
          // int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
          // int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

          // Rect object((int)xLeftBottom, (int)yLeftBottom,
          //             (int)(xRightTop - xLeftBottom),
          //             (int)(yRightTop - yLeftBottom));

          // object = object  & cv::Rect(0, 0, depth_mat.cols, depth_mat.rows);

          // Calculate mean depth inside the detection region
          // This is a very naive way to estimate objects depth
          // but it is intended to demonstrate how one might 
          // use depht data in general
          // Scalar m = mean(depth_mat(object));

          // std::ostringstream ss;
          // ss << classNames[objectClass] << " ";
          // ss << std::setprecision(2) << m[0] << " meters away";
          // String conf(ss.str());

          // rectangle(color_mat, object, Scalar(0, 255, 0));
          // int baseLine = 0;
          // Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

          // auto center = (object.br() + object.tl())*0.5;
          // center.x = center.x - labelSize.width / 2;

          // rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
          //     Size(labelSize.width, labelSize.height + baseLine)),
          //     Scalar(255, 255, 255), CV_FILLED);
          // putText(color_mat, ss.str(), center,
          //         FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
      }
  }

    win.beginPaint();
    glfw.draw2x2Streams(win.window, 2,
        null, 'rgb8', null, null,
        color.data, 'rgb8', color.width, color.height);
    win.endPaint();

  // Build the color map
  // const depthMap = colorizer.colorize(frameset.depthFrame);
  // if (depthMap) {
  //   // Paint the images onto the window
  //   win.beginPaint();
  //   const color = frameset.colorFrame;
  //   glfw.draw2x2Streams(win.window, 2,
  //       depthMap.data, 'rgb8', depthMap.width, depthMap.height,
  //       color.data, 'rgb8', color.width, color.height);
  //   win.endPaint();
  // }
}

pipeline.stop();
pipeline.destroy();
win.destroy();
rs2.cleanup();

