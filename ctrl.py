import numpy as np
import logging
import time

from main import main

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Control gains adapted from droneControl's DroneCommandController
YAW_GAIN = -0.05 # error_x (px) -> yaw_rate (deg/s)
UP_DOWN_GAIN = -0.05 # error_y (px) -> vertical_vel (cm/s)
FORWARD_GAIN = 0.001 # area_error (px^2) -> forward_vel (cm/s)
TARGET_AREA_RATIO = 0.08 # target bbox area as fraction of frame area
VERTICAL_SETPOINT = 0.45 # normalized y setpoint (0=top, 1=bottom)

MAX_YAW_RATE = 30 # deg/s
MAX_VERTICAL_VEL = 40 # cm/s
MAX_FORWARD_VEL = 40 # cm/s
DEADBAND_PX = 20 # px, suppress commands when error is small

# State machine timeouts
HOVER_TIMEOUT = 4.0 # seconds in HOVERING before RETURNING
RETURN_HOLD_DURATION = 2.0 # seconds holding before back to MANUAL
CONTROL_RATE = 0.1 # seconds between control loop ticks (10 Hz)

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

def compute_velocity_commands(target_data, frame_width, frame_height):
    bbox = target_data.get('bbox')
    if bbox is None:
        return 0, 0, 0, "HOVER (no bbox)!"
    
    x1, x2, y1, y2 = bbox
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    
    frame_cx = frame_width / 2
    frame_cy = frame_height / 2
    
    target_area = frame_width * frame_height * TARGET_AREA_RATIO
    
    ex = cx - frame_cx
    ey = cy - frame_cy
    
    ea = target_area - area
    
    if abs(ex) < DEADBAND_PX: ex = 0
    if abs(ey) < DEADBAND_PX: ey = 0
    
    yaw  = float(np.clip(YAW_GAIN * ex, -MAX_YAW_RATE, MAX_YAW_RATE))
    vert = float(np.clip(UP_DOWN_GAIN * ey, -MAX_VERTICAL_VEL, MAX_VERTICAL_VEL))
    fwd  = float(np.clip(FORWARD_GAIN * ea, -MAX_FORWARD_VEL, MAX_FORWARD_VEL))

    cmd_str = f"FWD={fwd:.0f}cm/s | VERT={vert:.0f}cm/s | YAW={yaw:.0f}°/s"
    return int(fwd), int(vert), int(yaw), cmd_str

def run_control_loop(tello, app_state):
    logging.info("Control loop started.")

    while not app_state.stop_event.is_set() and app_state.control_enabled.is_set():
        try:
            with app_state.state_lock:
                state = app_state.drone_state
            with app_state.target_lock:
                target_id = app_state.target_id
            with app_state.tracker_lock:
                target_data = app_state.tracked.get(target_id) if target_id else None
            with app_state.frame_lock:
                frame = app_state.frame

            fw = FRAME_WIDTH
            fh = FRAME_HEIGHT

            if state == "MANUAL":
                if app_state.airborne:
                    logging.info("Returning control to mobile app...")
                    logging.info("Wait at least five (5) seconds before resuming tracking.")
                    
                    wait = 50
                    
                    while (wait >= 0):
                        tello.send_rc_control()
                        time.sleep(0.1)
                        print('*')
                        wait -= 1
                    
                    resume = ''
                    while (resume.lower() != 'r'):
                        resume = input("Enter 'R' to resume: ")
                    resume_control(tello, app_state)

            elif state == "TRACKING":
                if target_id is None:
                    with app_state.state_lock:
                        tello.send_rc_control(0, 0, 0, 0)
                        app_state.drone_state = "MANUAL"
                    logging.info("[TRACKING] -> [MANUAL]: target cleared.")

                elif target_data is None:
                    with app_state.state_lock:
                        app_state.drone_state = "HOVERING"
                        app_state.hover_lost_time = time.time()
                    logging.info("[TRACKING] → [HOVERING]: target lost.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)
                else:
                    fwd, vert, yaw, cmd_str = compute_velocity_commands(target_data, fw, fh)
                    logging.debug(f"[TRACKING] {cmd_str}")
                    if app_state.airborne:
                        # send_rc_control(left_right, fwd_back, up_down, yaw)
                        tello.send_rc_control(0, fwd, vert, yaw)

            elif state == "HOVERING":
                elapsed = time.time() - app_state.hover_lost_time
                remaining = HOVER_TIMEOUT - elapsed

                if target_data is not None:
                    with app_state.state_lock:
                        app_state.drone_state = "TRACKING"
                    app_state.hover_lost_time = None
                    logging.info("[HOVERING] -> [TRACKING]: target reacquired.")

                elif elapsed >= HOVER_TIMEOUT:
                    with app_state.state_lock:
                        app_state.drone_state = "RETURNING"
                        app_state.hold_start_time = time.time()
                    logging.info("[HOVERING] -> [RETURNING]: timeout.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)
                else:
                    logging.debug(f"[HOVERING] {remaining:.1f}s remaining.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)

            elif state == "RETURNING":
                elapsed = time.time() - app_state.hold_start_time
                if elapsed >= RETURN_HOLD_DURATION:
                    with app_state.state_lock:
                        app_state.drone_state = "MANUAL"
                    with app_state.target_lock:
                        app_state.target_id = None
                    app_state.hold_start_time = None
                    logging.info("[RETURNING] -> [MANUAL].")

        except Exception as e:
            if not app_state.stop_event.is_set():
                logging.error(f"Control loop error: {e}")

        time.sleep(CONTROL_RATE)

    logging.info("Control loop stopped.")
    
def resume_control(tello, app_state):
        main()
        