#!/usr/bin/env python3
# Use semanantic segmentation to determine simple path tracking for a vehicle

# Based on https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/segnet.py

# May require "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" in ~/.bashrc
# Use cmake -DENABLE_NVMM=off in making jetson_inference

# https://github.com/dusty-nv/jetson-inference/issues/1493

# RTP at GCS:
# gst-launch-1.0 udpsrc port=5400 caps='application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264' ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
# ./segmav.py --headless --input-codec=H264 file://record-20230422-145739.mp4

import sys
import argparse
import os
import numpy as np
import cv2  # pylint: disable=import-error
import threading
import time
import signal
from datetime import datetime

from jetson_inference import segNet  # pylint: disable=import-error
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize, cudaAllocMapped, cudaToNumpy, cudaDrawCircle, cudaDrawLine, cudaDrawRect, cudaFont  # pylint: disable=import-error

exit_event = threading.Event()


def signal_handler(signum, frame):
    exit_event.set()


def ComputeStats(net, grid_width, grid_height, class_mask_np, num_classes):

    # compute the number of times each class occurs in the mask
    class_histogram, _ = np.histogram(
        class_mask_np, bins=num_classes, range=(0, num_classes-1))

    print('grid size:   {:d}x{:d}'.format(grid_width, grid_height))
    print('num classes: {:d}'.format(num_classes))

    print('-----------------------------------------')
    print(' ID  class name        count     %')
    print('-----------------------------------------')

    for n in range(num_classes):
        percentage = (
            float(class_histogram[n]) / float(grid_width * grid_height))*100
        print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(
            n, net.GetClassDesc(n), class_histogram[n], percentage))


class VideoThread(threading.Thread):
    '''
    Thread to record video streams
    '''

    def __init__(self, args, aargv, filename):
        threading.Thread.__init__(self)
        self.should_exit = False
        self.input = videoSource(args.input, argv=aargv)
        self.output = videoOutput(
            "file://{0}".format(filename), argv=aargv+["--headless"])

    def exit(self):
        self.should_exit = True

    def run(self):
        print("Starting Video thread")
        while True:
            # capture the next image
            img_input = self.input.Capture()

            if img_input is None:  # timeout
                continue

            # render the output image
            self.output.Render(img_input)

            cudaDeviceSynchronize()

            # exit on input/output EOS
            if not self.input.IsStreaming() or not self.output.IsStreaming():
                break

            if self.should_exit:
                break
        print("Exiting Video thread")


class SegThread(threading.Thread):
    '''
    Thread to segment video streams
    '''

    def __init__(self, args, aargv, is_headless):
        threading.Thread.__init__(self)
        self.should_exit = False
        self.is_headless = is_headless
        self.args = args
        # load the segmentation network
        self.net = segNet(args.network, [])

        # set the alpha blending value
        self.net.SetOverlayAlpha(args.alpha)

        # create video output. Add datetime if using file
        if args.output.startswith("file://"):
            name, ext = os.path.splitext(args.output)
            filename = "{0}-{1}{2}".format(name,
                                           datetime.now().strftime("%Y%m%d-%H%M%S"), ext)
            print(filename)
            self.output = videoOutput(filename, argv=aargv+["--headless"])
        else:
            self.output = videoOutput(args.output, argv=aargv)

        self.overlay = None
        self.class_mask = None
        self.class_mask_np = None
        self.grid_width = None
        self.grid_height = None

        self.numAveraging = 3

        self.font = cudaFont()

        # create video source
        self.input = videoSource(args.input, argv=aargv)

        # bearings history
        self.bearings = []
        self.threadLock = threading.Lock()
        self.timeOfLastUpdate = 0

        # latency checks
        self.timeOfCapture = []

    def exit(self):
        self.should_exit = True

    def run(self):
        print("Starting Segment thread")
        while True:
            # capture the next image
            img_input = self.input.Capture()
            with self.threadLock:
                self.timeOfCapture.append(time.time())
                if len(self.timeOfCapture) > self.numAveraging:
                    self.timeOfCapture = self.timeOfCapture[-self.numAveraging:]

            if img_input is None:  # timeout
                continue

            # allocate buffers for this size image, if not already allocated
            if not self.overlay:
                self.overlay = cudaAllocMapped(width=img_input.shape[1],
                                               height=img_input.shape[0],
                                               format=img_input.format)
            if not self.class_mask:
                grid_width, grid_height = self.net.GetGridSize()
                self.class_mask = cudaAllocMapped(
                    width=grid_width, height=grid_height, format="gray8")
                self.class_mask_np = cudaToNumpy(self.class_mask)

            # process the segmentation network
            self.net.Process(img_input, ignore_class=self.args.ignore_class)

            # generate the overlay
            self.net.Overlay(self.overlay, filter_mode=self.args.filter_mode)

            # print out performance info
            cudaDeviceSynchronize()

            # get the class mask (each pixel contains the classID for that grid cell)
            self.net.Mask(self.class_mask, grid_width, grid_height)

            # compute segmentation class grid and stats
            # ComputeStats(net, grid_width, grid_height, class_mask_np, net.GetNumClasses())

            # mask to target class
            mask = cv2.inRange(self.class_mask_np,  # pylint: disable=no-member
                               self.args.targetclass, self.args.targetclass)

            # zoom and blur
            scale_percent = 400  # percent of original size
            width = int(mask.shape[1] * scale_percent / 100)
            height = int(mask.shape[0] * scale_percent / 100)
            dim = (width, height)
            maskzoom = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member

            # kernel = np.ones((2, 2), np.uint8)
            # cv2.morphologyEx(maskzoom, cv2.MORPH_OPEN, kernel)
            maskzoomblur = maskzoom

            # get extents of object
            # find contours in the binary image
            contours, hierarchy = cv2.findContours(  # pylint: disable=no-member
                maskzoomblur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # pylint: disable=no-member

            # No contours found, go to next image
            if len(contours) == 0:
                print("No contours")
                self.output.Render(self.overlay)
                continue

            # get (and show) largest countour. Need to simplify contour a bit to reduce noise
            contourLargest = max(contours, key=cv2.contourArea)  # pylint: disable=no-member
            perimeter = cv2.arcLength(contourLargest, True)  # pylint: disable=no-member
            contourLargest = cv2.approxPolyDP(  # pylint: disable=no-member
                contourLargest, 0.03 * perimeter, True)
            for i in range(0, len(contourLargest)):
                curpoint = contourLargest[i-1][0]
                nextpoint = contourLargest[i][0]
                overlayX1 = int(
                    (curpoint[0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                overlayY1 = int(
                    (curpoint[1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                overlayX2 = int(
                    (nextpoint[0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                overlayY2 = int(
                    (nextpoint[1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                cudaDrawLine(self.overlay,
                             [overlayX1, overlayY1],
                             [overlayX2, overlayY2],
                             (255, 0, 0),
                             3)  # (x1,y1), (x2,y2), color, thickness

            # Split ROI into horizontal strips
            x1, y1, w, h = cv2.boundingRect(contourLargest)  # pylint: disable=no-member
            stripregions = []
            numregions = 2
            for i in range(numregions):
                stripregions.append(
                    ((0, y1 + int(i*(h/numregions))), (width, y1 + int((i+1)*(h/numregions)))))

            # and display the strips
            for strip in stripregions:
                # cv2.rectangle(maskzoomblur, strip[0], strip[1], (128, 128, 128), 1)
                overlayX1 = int(
                    (strip[0][0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                overlayY1 = int(
                    (strip[0][1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                overlayX2 = int(
                    (strip[1][0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                overlayY2 = int(
                    (strip[1][1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                cudaDrawRect(self.overlay,
                             (overlayX1, overlayY1, overlayX2, overlayY2),
                             (0, 0, 0, 0),
                             line_color=(0, 75, 255, 200))  # (left, top, right, bottom), color

            # get centroid of each strip
            centroids = []
            for strip in stripregions:
                M = cv2.moments(maskzoomblur[:][strip[0][1]:strip[1][1]])  # pylint: disable=no-member
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append(((cX, cY + strip[0][1])))
                # cv2.circle(maskzoomblur, (cX, cY + strip[0][1]), 2, (128, 128, 128), -1)
                # Video overlay
                overlayX = int((cX/(scale_percent/100)) *
                               (img_input.shape[1] / grid_width))
                overlayY = int(
                    ((cY + strip[0][1])/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                cudaDrawCircle(self.overlay, (overlayX, overlayY),
                               10, (255, 255, 255))

            # do a bit of quality control. Vector len should be less than 20 and bearing +-45deg
            # Vector is the bottom two centroids
            lenvec = np.linalg.norm(
                np.array(centroids[1]) - np.array(centroids[0]))
            bearing_rel = np.arctan(
                (centroids[1][0] - centroids[0][0]) / (centroids[1][1] - centroids[0][1]))
            # print("Len={0:.0f}, Bearing={1:.1f}".format(lenvec, np.rad2deg(bearing_rel)))
            # print("Grid is " + str(self.net.GetGridSize()))
            # print("Thres is " + str(1.6 * self.net.GetGridSize()[1]))
            if lenvec < 2*self.net.GetGridSize()[1] and bearing_rel == np.clip(bearing_rel, -np.pi/4, np.pi/4):

                # and draw line between centroids
                for i in range(len(centroids)):
                    # cv2.line(maskzoomblur, centroids[i], centroids[i+1], (128, 128, 128), 2)
                    # Video overlay
                    overlayX1 = int(
                        (centroids[i-1][0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                    overlayY1 = int(
                        (centroids[i-1][1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                    overlayX2 = int(
                        (centroids[i][0]/(scale_percent/100)) * (img_input.shape[1] / grid_width))
                    overlayY2 = int(
                        (centroids[i][1]/(scale_percent/100)) * (img_input.shape[0] / grid_height))
                    cudaDrawLine(self.overlay,
                                 [overlayX1, overlayY1],
                                 [overlayX2, overlayY2],
                                 (255, 255, 255),
                                 4)  # (x1,y1), (x2,y2), color, thickness

                # Then average to smooth out over the last self.numAveraging readings.
                with self.threadLock:
                    self.bearings.append(bearing_rel)
                    if len(self.bearings) > self.numAveraging:
                        self.bearings = self.bearings[-self.numAveraging:]
                    self.timeOfLastUpdate = time.time()
            else:
                print("Contour not strong enough to get bearing")

            # Show debug window in OpenCV
            # if not self.is_headless:
            #     cv2.imshow('image', maskzoomblur)
            #     k = cv2.waitKey(10) & 0XFF
            #     if k == 27:
            #         break
            if self.getBearing() != None:
                bearingstr = "Rel bearing: {0:.1f} deg".format(
                    self.getBearing())
            else:
                bearingstr = "Rel bearing: N/A"
            self.font.OverlayText(self.overlay,
                                  img_input.shape[1],
                                  img_input.shape[0],
                                  bearingstr,
                                  5, 5,
                                  self.font.White, self.font.Gray40)

            # render the output image
            self.output.Render(self.overlay)

            # update the title bar
            self.output.SetStatus("{:s} | Network {:.0f} FPS".format(
                self.args.network, self.net.GetNetworkFPS()))

            # exit on input/output EOS
            if not self.input.IsStreaming() or not self.output.IsStreaming():
                if not self.is_headless:
                    cv2.destroyAllWindows()  # pylint: disable=no-member
                break

            if self.should_exit:
                break
        print("Exiting segment thread")

    '''
    Get current calculated bearing in degrees. return None is there's been no update in 2 sec
    -ve is ccw direction, +ve is cw direction
    '''

    def getBearing(self):
        with self.threadLock:
            if time.time() - self.timeOfLastUpdate > 2:
                print("No bearing in {0:.2f} sec".format(
                    time.time() - self.timeOfLastUpdate))
                return None
            avg_bearing = -0.7 * np.mean(self.bearings)
            print("Rel bearing is {0:.2f}deg".format(np.rad2deg(avg_bearing)))
            # print(self.bearings)
            return np.rad2deg(avg_bearing)

    '''
    Get the average latency of the image capturing and processing
    '''

    def getLatency(self):
        with self.threadLock:
            avg_latency = time.time() - np.mean(self.timeOfCapture)
            print("Latency is {0:.0f} millisec".format(avg_latency * 1000))


if __name__ == '__main__':
    # parse the command line
    parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage())

    parser.add_argument("input", type=str, default="csi://0",
                        nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="rtp://192.168.1.124:5400",
                        nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str,
                        default="fcn-resnet18-cityscapes-1024x512", help="pre-trained model to load")
    parser.add_argument("--filter-mode", type=str, default="point", choices=[
                        "point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'point')")
    parser.add_argument("--ignore-class", type=str, default="void",
                        help="optional name of class to ignore in the visualization results (default: 'void')")
    parser.add_argument("--alpha", type=float, default=80.0,
                        help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
    parser.add_argument("--targetclass", type=int, default=3,
                        help="The item class to track")

    is_headless = [
        "--headless"] if sys.argv[0].find('console.py') != -1 else [""]

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start segmentation thread
    curThread = SegThread(args, sys.argv, is_headless)
    curThread.start()
    while True:
        if exit_event.is_set():
            if curThread:
                curThread.should_exit = True
            break
        time.sleep(0.5)
        curThread.getBearing()
        curThread.getLatency()
    time.sleep(1)
    print("-----Exited-----")
