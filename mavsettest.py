#!/usr/bin/env python3
# Test script for send velocity and yaw commands in GUIDED mode to rover
# Using defaults:
# When RC9 is set to 2000, the rover will run at 1m/s and 5 degrees relative yaw
# When RC9 is set to 1000, the rover will stop
# The vehicle does need to be armd beforehand

import argparse
from pymavlink import mavutil
import threading
import signal
import sys
import time

import numpy as np

exit_event = threading.Event()


def signal_handler(signum, frame):
    exit_event.set()


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
    parser = argparse.ArgumentParser(
        description="Test for send vel and yaw to Rover")
    parser.add_argument(
        "--device", type=str, default="udpin:127.0.0.1:14550", help="MAVLink connection string")
    parser.add_argument("--baud", type=int, default=115200,
                        help="MAVLink baud rate, if using serial")
    parser.add_argument("--source-system", type=int,
                        default=1, help="MAVLink Source system")
    parser.add_argument("--rc", type=int, default=9,
                        help="2-pos RC Channel to control script")
    parser.add_argument("--vel", type=float, default=1,
                        help="Forward velocity setpoint")
    parser.add_argument("--pwmlow", type=int, default=1000,
                        help="RC PWM low value")
    parser.add_argument("--pwmhigh", type=int, default=2000,
                        help="RC PWM high value")
    try:
        args = parser.parse_known_args()[0]
    except:
        parser.print_help()
        sys.exit(0)

    time_of_last_bearing_check = 0
    set_guided = 0

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
    curHeading = -1

    while True:
        msg = conn.recv_match(blocking=True, timeout=0.5)
        if msg:
            if msg.get_type() == 'VFR_HUD':
                curHeading = msg.heading
            if msg.get_type() == 'RC_CHANNELS':
                if not rc_level:
                    rc_level = getattr(msg, fieldname)
                if abs(rc_level - getattr(msg, fieldname)) > 100:
                    # Has the level changed?
                    if abs(args.pwmlow - getattr(msg, fieldname)) < 50:
                        set_target(conn, 0, 0)
                        print("Stopped NAV")
                        time_of_last_bearing_check = 0
                    if abs(args.pwmhigh - getattr(msg, fieldname)) < 50:
                        # Start Nav
                        print("Started NAV")
                        conn.set_mode("GUIDED")
                        time_of_last_bearing_check = time.time() - 3
                    rc_level = getattr(msg, fieldname)
        if exit_event.is_set():
            break
        if time_of_last_bearing_check > 0 and time.time() - time_of_last_bearing_check > 2:
            print(conn.flightmode)
            # Check every 2 sec, if running nav routine (ie time_of_last_bearing_check > 0)
            bearing = 5  # hard set to 5 degrees rel yaw for this example
            time_of_last_bearing_check = time.time()
            if curHeading > 0:
                target_bearing = (curHeading + bearing) % 360
            else:
                target_bearing = -1
            print("Rel is {0:.0f} deg, cur is {2:.0f}, Target is {1:.0f}".format(
                bearing, target_bearing, curHeading))
            # send command to rover
            set_target(conn, args.vel, np.deg2rad(bearing))

    print("-----Exiting-----")
