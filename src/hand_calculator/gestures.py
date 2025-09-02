"""
Gesture recognition and mapping.
Converts finger positions to calculator tokens.
"""

import json
import os
import time

class GestureRecognizer:
    def __init__(self, stability_frames=15, cooldown_time=1.0):
        self.stability_frames = stability_frames
        self.cooldown_time = cooldown_time
        
        self.current_gesture = None
        self.gesture_count = 0
        self.last_token_time = 0
        self.last_stable_gesture = None
        
        self.gesture_map = self._load_gesture_map()
    
    def _load_gesture_map(self):
        """Load gesture mapping from JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'gesture_map.json')
        
        # Default mapping - all keys MUST be strings with quotes
        default_map = {
            "(0, 0, 0, 0, 0)": "0",
            "(0, 1, 0, 0, 0)": "1",
            "(0, 1, 1, 0, 0)": "2",
            "(0, 1, 1, 1, 0)": "3",
            "(0, 1, 1, 1, 1)": "4",
            "(1, 1, 1, 1, 1)": "5",
            "(1, 0, 1, 1, 1)": "6",
            "(1, 1, 0, 1, 1)": "7",
            "(1, 1, 1, 0, 1)": "8",
            "(1, 1, 1, 1, 0)": "9",
            "(1, 0, 0, 0, 0)": "+",
            "(1, 1, 0, 0, 0)": "-",
            "(0, 0, 1, 1, 1)": "*",
            "(0, 0, 0, 1, 1)": "/",
            "(0, 0, 0, 0, 1)": "=",
            "(1, 0, 0, 0, 1)": "C",
            "(0, 0, 1, 0, 0)": "⌫"
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_map, f, indent=2)
                return default_map
        except Exception as e:
            print(f"Error loading gesture map: {e}, using default")
            return default_map
    
    def fingers_up(self, landmarks):
        """Determine which fingers are up based on landmarks."""
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Thumb (compare x-coordinates for horizontal orientation)
        if landmarks[4][0] > landmarks[3][0]:  # Right hand
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers (compare y-coordinates)
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:  # Tip higher than PIP
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture from hand landmarks."""
        if not landmarks:
            return None
            
        fingers = self.fingers_up(landmarks)
        gesture_key = str(tuple(fingers))
        
        return self.gesture_map.get(gesture_key, None)
    
    def update_gesture_state(self, current_gesture):
        """Update gesture state with stability checking and cooldown."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_token_time < self.cooldown_time:
            return None
        
        # Update gesture tracking
        if current_gesture == self.current_gesture:
            self.gesture_count += 1
        else:
            self.current_gesture = current_gesture
            self.gesture_count = 1
        
        # Check if gesture is stable
        if (self.gesture_count >= self.stability_frames and 
            current_gesture and 
            current_gesture != self.last_stable_gesture):
            
            self.last_stable_gesture = current_gesture
            self.last_token_time = current_time
            self.gesture_count = 0
            return current_gesture
        
        return None
    
    def get_stability_info(self):
        """Get current stability information for UI display."""
        progress = min(self.gesture_count / self.stability_frames, 1.0) if self.current_gesture else 0
        return {
            'current_gesture': self.current_gesture,
            'progress': progress,
            'count': self.gesture_count,
            'required': self.stability_frames
        }
