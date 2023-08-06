#!/usr/bin/env python3
# This is a generic script for accessing MAVLink data over UDP port 14552

import argparse
from pymavlink import mavutil
from datetime import datetime
import threading
import signal
import sys
import time
from jetson_inference import segNet  # pylint: disable=import-error
from jetson_utils import videoSource, videoOutput  # pylint: disable=import-error
import numpy as np

from segmav import SegThread, VideoThread
exit_event = threading.Event()


def signal_handler(signum, frame):
    exit_event.set()


# https://mavlink.io/en/messages/common.html#STATUSTEXT
def send_msg_to_gcs(conn, text_to_be_sent):
    '''
    Send a text message to the GCS console
    '''
    # MAV_SEVERITY: 0=EMERGENCY 1=ALERT 2=CRITICAL 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG, 8=ENUM_END
    text_msg = 'SEGMAV: ' + text_to_be_sent
    conn.mav.statustext_send(
        mavutil.mavlink.MAV_SEVERITY_INFO, text_msg.encode())


# https://mavlink.io/en/messages/common.html#PLAY_TUNE
def play_tune(conn, tune):
    '''
    Tell the flight controller to play a tune on the buzzer, if fitted
    '''
    conn.mav.play_tune_send(
        conn.target_system, conn.target_component, bytes(tune, "ascii"))


# https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED
def set_target(conn, speed, yaw):
    '''
    Set the (relative) direction and speed of travel in guided mode
    '''
    POSITION_TARGET_TYPEMASK = 0b101111000111  # Only target vel and yaw
    conn.mav.set_position_target_local_ned_send(0,
                                                conn.target_system,
                                                conn.target_component,
                                                mavutil.mavlink.MAV_FRAME_BODY_NED,
                                                POSITION_TARGET_TYPEMASK,
                                                0, 0, 0,
                                                speed, 0, 0,
                                                0, 0, 0,
                                                yaw, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation-based navigation for Ardupilot",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage())
    parser.add_argument("input", type=str, default="csi://0",
                        nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="rtp://192.168.1.124:5400",
                        nargs='?', help="URI of the output stream")
    parser.add_argument(
        "--device", type=str, default="udpin:127.0.0.1:14550", help="MAVLink connection string")
    parser.add_argument("--baud", type=int, default=115200,
                        help="MAVLink baud rate, if using serial")
    parser.add_argument("--source-system", type=int,
                        default=1, help="MAVLink Source system")
    parser.add_argument("--rc", type=int, default=10,
                        help="3-pos RC Channel to control script")
    parser.add_argument("--pwmlow", type=int, default=1000,
                        help="RC PWM low value")
    parser.add_argument("--pwmmid", type=int, default=1500,
                        help="RC PWM mid value")
    parser.add_argument("--pwmhigh", type=int, default=2000,
                        help="RC PWM high value")
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
    parser.add_argument("--vel", type=float, default=0.6,
                        help="Forward velocity setpoint")

    is_headless = [
        "--headless"] if sys.argv[0].find('console.py') != -1 else [""]

    time_of_last_bearing_check = 0

    try:
        args = parser.parse_known_args()[0]
    except:
        parser.print_help()
        sys.exit(0)

    curThread = None

    signal.signal(signal.SIGINT, signal_handler)

    # Setup MAVLink to connect
    conn = mavutil.mavlink_connection(args.device, autoreconnect=True, source_system=args.source_system,
                                      baud=args.baud, force_connected=False,
                                      source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)

    # wait for the heartbeat msg to find the system ID
    print("Waiting for Hearbeat from ArduPilot")
    while True:
        m = conn.wait_heartbeat(timeout=0.5)
        if m is not None and m.type not in [mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_TYPE_GENERIC]:
            # Got a heartbeat from remote MAVLink device that's NOT a GCS, good to continue
            break
        if exit_event.is_set():
            print("Exiting main")
            sys.exit(0)

    print("Got Heartbeat from ArduPilot (system {0} component {1})".format(conn.target_system,
                                                                           conn.target_system))

    rc_level = None
    fieldname = "chan" + str(args.rc) + "_raw"

    print("Monitoring RC channel {0}".format(args.rc))
    send_msg_to_gcs(conn, "Monitoring RC channel {0}".format(args.rc))
    curHeading = -1

    while True:
        msg = conn.recv_match(blocking=True, timeout=0.1)
        if msg:
            if msg.get_type() == 'VFR_HUD':
                curHeading = msg.heading
            if msg.get_type() == 'RC_CHANNELS':
                if not rc_level:
                    rc_level = getattr(msg, fieldname)
                if abs(rc_level - getattr(msg, fieldname)) > 100:
                    # Has the level changed?
                    if abs(args.pwmlow - getattr(msg, fieldname)) < 50:
                        if curThread:
                            curThread.exit()
                            time.sleep(2)
                            curThread = None
                        send_msg_to_gcs(conn, "Stopped Record/NAV")
                        time_of_last_bearing_check = 0
                        play_tune(conn, "L12DD")  # two fast medium tones
                    if abs(args.pwmmid - getattr(msg, fieldname)) < 50:
                        if not curThread:
                            print("RC{0} changed to {1}. Doing RECORD action".format(
                                args.rc, getattr(msg, fieldname)))
                            filename = "record-{0}.mp4".format(
                                datetime.now().strftime("%Y%m%d-%H%M%S"))
                            print("Filename is {0}".format(filename))
                            time_of_last_bearing_check = 0
                            try:
                                # Start Record thread
                                curThread = VideoThread(
                                    args, sys.argv, filename)
                                curThread.start()
                                send_msg_to_gcs(
                                    conn, "Started {0}".format(filename))
                                # two fast medium tones
                                play_tune(conn, "L12DD")
                            except Exception as ex:
                                print(ex)
                                # three very fast, high tones
                                play_tune(conn, "L16FFF")
                        else:
                            print("Can't start recording. Process still active")
                            send_msg_to_gcs(
                                conn, "Can't start. Record still active")
                            # three very fast, high tones
                            play_tune(conn, "L16FFF")
                    if abs(args.pwmhigh - getattr(msg, fieldname)) < 50:
                        if not curThread:
                            # Start Nav thread
                            curThread = SegThread(args, sys.argv, is_headless)
                            curThread.start()
                            print("RC{0} changed to {1}. Doing SEGMAV action".format(
                                args.rc, getattr(msg, fieldname)))
                            send_msg_to_gcs(conn, "Started NAV")
                            conn.set_mode("GUIDED")
                            time_of_last_bearing_check = time.time()
                            play_tune(conn, "L12DD")  # two fast medium tones
                        else:
                            print(
                                "Can't start SEGMAV. Recording process still active")
                            send_msg_to_gcs(
                                conn, "Can't start. NAV still active")
                            # three very fast, high tones
                            play_tune(conn, "L16FFF")
                    rc_level = getattr(msg, fieldname)
        if exit_event.is_set():
            if curThread:
                curThread.exit()
            break
        if time_of_last_bearing_check > 0 and time.time() - time_of_last_bearing_check > 0.3 and curThread:
            # Check every 0.3 sec, if running nav routine (ie time_of_last_bearing_check > 0)
            bearing = curThread.getBearing()
            time_of_last_bearing_check = time.time()
            if bearing is not None:
                if curHeading > 0:
                    target_bearing = (curHeading + bearing) % 360
                else:
                    target_bearing = -1
                send_msg_to_gcs(conn, "Rel is {0:.0f} deg, cur is {2:.0f}, Target is {1:.0f}".format(
                    bearing, target_bearing, curHeading))
                # send command to rover. Will only affect in GUIDED mode set_position_target_local_ned
                set_target(conn, args.vel, np.deg2rad(bearing))

    time.sleep(2)
    print("-----Exiting-----")
    send_msg_to_gcs(conn, "Exiting SEGMAV")
