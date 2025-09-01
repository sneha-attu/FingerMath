"""
Hand tracking using MediaPipe.
Provides hand landmark detection and utilities.
"""

import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def process_frame(self, frame):
        """Process frame and return hand landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list.append(self._extract_landmarks(hand_landmarks, frame.shape))
                
        return landmarks_list
    
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """Extract normalized landmarks from MediaPipe results."""
        landmarks = []
        h, w = frame_shape[:2]
        
        for lm in hand_landmarks.landmark:
            landmarks.append([int(lm.x * w), int(lm.y * h)])
            
        return landmarks
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = tuple(landmarks[start_idx])
                end_point = tuple(landmarks[end_idx])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x, y = landmark
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            else:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    def get_finger_positions(self, landmarks):
        """Get finger tip and PIP positions for gesture recognition."""
        if len(landmarks) < 21:
            return None
            
        finger_positions = {
            'thumb': {'tip': landmarks[4], 'pip': landmarks[3]},
            'index': {'tip': landmarks[8], 'pip': landmarks[6]},
            'middle': {'tip': landmarks[12], 'pip': landmarks[10]},
            'ring': {'tip': landmarks[16], 'pip': landmarks[14]},
            'pinky': {'tip': landmarks[20], 'pip': landmarks[18]}
        }
        
        return finger_positions
