import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Control gains adapted from droneControl's DroneCommandController
YAW_GAIN = 1.0 # error_x (px) -> yaw_rate (deg/s)
UP_DOWN_GAIN = 2.0 # error_y (px) -> vertical_vel (cm/s)
FORWARD_GAIN = 1.0 # area_error (px^2) -> forward_vel (cm/s)
TARGET_AREA_RATIO =  0.08 # target bbox area as fraction of frame area

MAX_YAW_RATE = 20 # deg/s
MAX_VERTICAL_VEL = 20 # cm/s
MAX_FORWARD_VEL = 20 # cm/s
DEADBAND_PX = 30 # px, suppress commands when error is small

# State machine timeouts
HOVER_TIMEOUT = 4.0 # seconds in HOVERING before RETURNING
RETURN_HOLD_DURATION = 2.0 # seconds holding before back to MANUAL
CONTROL_RATE = 0.1 # seconds between control loop ticks (10 Hz)

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

def compute_velocity_commands(target_data, frame_width, frame_height, app_state):
    bbox = target_data.get('bbox')
    if bbox is None:
        return 0, 0, 0, "HOVER (no bbox)!"
    
    x1, y1, x2, y2 = bbox
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    
    """ edge_proximity  = min(cx, frame_width - cx, cy, frame_height - cy) # Distance to nearest edge of the frame
    if edge_proximity > app_state.tracker.edge_margin:
        return (0, 0, 0, "Edge margin threshold exceeded.") """
    
    frame_cx = frame_width / 2
    frame_cy = frame_height / 2
    
    target_area = frame_width * frame_height * TARGET_AREA_RATIO
    
    ex = cx - frame_cx
    ey = cy - frame_cy
    
    ea = target_area - area
    
    if abs(ex) < DEADBAND_PX: ex = 0
    if abs(ey) < DEADBAND_PX: ey = 0
    
    ex_norm = ex / (frame_width / 2)
    ey_norm = ey / (frame_height / 2)
    ea_norm = ea / (frame_width * frame_height)
    
    yaw  = float(np.clip(MAX_YAW_RATE * YAW_GAIN * ex_norm, -MAX_YAW_RATE, MAX_YAW_RATE))
    vert = float(np.clip(MAX_VERTICAL_VEL * UP_DOWN_GAIN * ey_norm, -MAX_VERTICAL_VEL, MAX_VERTICAL_VEL))
    fwd  = float(np.clip(MAX_FORWARD_VEL * FORWARD_GAIN * ea_norm, -MAX_FORWARD_VEL, MAX_FORWARD_VEL))
    
    logging.info(f"ex_norm={ex_norm:.3f} ey_norm={ey_norm:.3f} ea_norm={ea_norm:.3f} | "
    f"FWD={fwd} VERT={vert} YAW={yaw}")
    
    cmd_str = f"FWD={fwd:.0f}cm/s | VERT={vert:.0f}cm/s | YAW={yaw:.0f}°/s"
    return round(fwd), round(vert), round(yaw), cmd_str

def run_control_loop(tello, app_state):
    logging.info("Control loop started.")
    
    control_runtime_ds = 0 # control loop runtime in deciseconds

    while not app_state.stop_event.is_set():
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
                    tello.send_rc_control(0, 0, 0, 0)

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
                    fwd, vert, yaw, cmd_str = compute_velocity_commands(target_data, fw, fh, app_state)
                    if control_runtime_ds % 50 == 0: # log command strings every five seconds
                        logging.info(f"[TRACKING] {cmd_str}")
                    if app_state.airborne:
                        # send_rc_control(left_right, fwd_back, up_down, yaw)
                        tello.send_rc_control(0, fwd, vert, yaw)

            elif state == "HOVERING":
                elapsed   = time.time() - app_state.hover_lost_time
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
                    
            control_runtime_ds += 1 # Add 1 to running time

        except Exception as e:
            if not app_state.stop_event.is_set():
                logging.error(f"Control loop error: {e}")

        time.sleep(CONTROL_RATE)

    logging.info("Control loop stopped.")