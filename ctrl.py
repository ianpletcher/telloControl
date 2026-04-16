import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Control gains adapted from droneControl's DroneCommandController
YAW_GAIN = 1.0 # error_x (px) -> yaw_rate (deg/s)
UP_DOWN_GAIN = 2.0 # error_y (px) -> vertical_vel (cm/s)
FORWARD_GAIN = 1.0 # area_error (px^2) -> forward_vel (cm/s)
GOAL_AREA_RATIO =  0.08 # target bbox area as fraction of frame area

MAX_YAW_RATE = 20 # deg/s
MAX_VERTICAL_VEL = 20 # cm/s
MAX_FORWARD_VEL = 20 # cm/s
DEADBAND_PX = 20 # px, suppress commands when error is small

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
    
    x1, y1, x2, y2 = bbox # left, top, right, bottom
    
    cx = (x1 + x2) / 2 # target centroid x coordinate
    cy = (y1 + y2) / 2 # target centroid y coordinate
    area = (x2 - x1) * (y2 - y1) # target bounding box area
    
    frame_cx = frame_width / 2 # frame centroid x coordinate
    frame_cy = frame_height / 2 # frame centroid y coordinate
    
    goal_area = frame_width * frame_height * GOAL_AREA_RATIO # Desired area of target as percentage of frame
    
    ex = cx - frame_cx # Pixel-distance between frame centroid and target centroid in x direction
    ey = cy - frame_cy # Pixel-distance between frame centroid and target centroid in y direction
    
    ea = goal_area - area # Pixel-area difference between goal area and actual target bbox area
    
    edge_proximity_x = min(cx, frame_width - cx) # Distance from target centroid to closest vertical edge
    edge_proximity_y = min(cy, frame_height - cy) # Distance from target centroid to closest horizontal edge
    
    if edge_proximity_x < DEADBAND_PX: # If target is close to vertical edge, suppress x error to avoid erratic yaw commands
        ex = 0
    elif edge_proximity_y < DEADBAND_PX: # If target is close to horizontal edge, suppress y error to avoid erratic vertical commands
        ey = 0
    elif ea < 0 and (edge_proximity_x < DEADBAND_PX * 2 or edge_proximity_y < DEADBAND_PX * 2): # If target is larger than goal area and close to any edge, suppress area error to avoid aggressive forward commands
        ea = 0
    
        
    if abs(ex) < DEADBAND_PX: ex = 0 # Issue a 0 command in this direction if error is small enough, reducing erratic behavior
    if abs(ey) < DEADBAND_PX: ey = 0 # Issue a 0 command in this direction if error is small enough, reducing erratic behavior
    
    ex_norm = ex / (frame_width / 2) # Normalized x error as a fraction of frame_width / 2, yielding a value in [-1, 1]
    ey_norm = ey / (frame_height / 2) # Normalized y error as a fraction of frame_height / 2, yielding a value in [-1, 1]
    ea_norm = ea / (frame_width * frame_height) # Normalized area error as a fraction of total frame area, yielding a value in [-1, 1]
    
    yaw  = float(np.clip(MAX_YAW_RATE * YAW_GAIN * ex_norm, -MAX_YAW_RATE, MAX_YAW_RATE)) # Yaw rate command in deg/s, clipped to max yaw rate
    vert = float(np.clip(MAX_VERTICAL_VEL * UP_DOWN_GAIN * ey_norm, -MAX_VERTICAL_VEL, MAX_VERTICAL_VEL)) # Vertical velocity command in cm/s, clipped to max vertical velocity
    fwd  = float(np.clip(MAX_FORWARD_VEL * FORWARD_GAIN * ea_norm, -MAX_FORWARD_VEL, MAX_FORWARD_VEL)) # Forward velocity command in cm/s, clipped to max forward velocity
    
    logging.info(f"ex_norm={ex_norm:.3f} ey_norm={ey_norm:.3f} ea_norm={ea_norm:.3f} | "
    f"FWD={fwd} VERT={vert} YAW={yaw}")
    
    cmd_str = f"FWD={fwd:.0f}cm/s | VERT={vert:.0f}cm/s | YAW={yaw:.0f}°/s"
    return round(fwd), round(vert), round(yaw), cmd_str # Return rounded integer commands for tello.send_rc_control, along with a string for logging

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