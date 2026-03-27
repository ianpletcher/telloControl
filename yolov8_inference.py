YOLO_CONFIDENCE = 0.35
YOLO_CLASSES = [2, 5, 7, 0]
MIN_BBOX_AREA = 2000

def run_yolo_inference(model, frame, app_state):
    results = model(frame, verbose=False, conf=YOLO_CONFIDENCE, classes=YOLO_CLASSES) # list of Results objects
    detections = []
    h, w = frame.shape[:2]
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) # Pixel coordinates from bounding box coordinate as a list
        conf = float(box.conf[0]) # Confidence score as float
        label = results[0].names[int(box.cls[0])] # Class label as string
        
        edge_exc = app_state.tracker.edge_margin # Exclude boxes too close to edges to improve tracking stability
        
        if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
            continue
        if (x1 < edge_exc or x2 > w - edge_exc or y1 < edge_exc or y2 > h - edge_exc):
            continue
        
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # 5×5 patch color sample
        py0, py1 = max(0, cy - 2), min(h, cy + 3)
        px0, px1 = max(0, cx - 2), min(w, cx + 3)
        color = frame[py0:py1, px0:px1].mean(axis=(0, 1)).tolist()
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'centroid': (cx, cy),
            'label': label,
            'confidence': conf,
            'color': color,
        })

    return detections
        