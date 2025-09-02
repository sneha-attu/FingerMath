"""
Enhanced hand tracking with multi-hand support.
"""

import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Updated to support 2 hands for multi-hand gestures
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,  # CHANGED: Now supports 2 hands
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
    
    def draw_landmarks(self, frame, landmarks, hand_index=0):
        """Draw hand landmarks on frame with different colors for each hand."""
        colors = [(0, 255, 0), (255, 0, 0)]  # Green for first hand, Red for second
        color = colors[hand_index % 2]
        
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
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x, y = landmark
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            else:
                cv2.circle(frame, (x, y), 5, color, -1)
