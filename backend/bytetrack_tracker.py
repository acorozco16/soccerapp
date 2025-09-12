import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
import cv2


@dataclass
class Detection:
    """Soccer ball detection with confidence"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # Soccer ball class


@dataclass
class Track:
    """Ball track with state"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    state: str  # 'Tracked', 'Lost', 'Removed'
    frame_id: int
    start_frame: int


class KalmanFilter:
    """Simple Kalman filter for ball tracking"""
    
    def __init__(self):
        self.dt = 1.0  # Time step
        # State: [x, y, vx, vy] - position and velocity
        self.x = np.zeros(4)  # State vector
        self.P = np.eye(4) * 1000  # Covariance matrix
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise - optimized for soccer ball movement
        self.Q = np.array([
            [4, 0, 0, 0],      # Position uncertainty
            [0, 4, 0, 0],
            [0, 0, 400, 0],    # Higher velocity uncertainty for fast ball
            [0, 0, 0, 400]
        ])
        
        # Measurement noise - lower for good ball detection
        self.R = np.array([
            [5, 0],    # Lower measurement noise
            [0, 5]
        ])
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]  # Return predicted position
    
    def update(self, measurement):
        """Update with measurement"""
        z = np.array(measurement)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class STrack:
    """Single track for ByteTrack"""
    
    shared_kalman = KalmanFilter()
    track_id = 0
    
    def __init__(self, bbox, confidence, frame_id):
        # Increment track ID
        STrack.track_id += 1
        self.track_id = STrack.track_id
        
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.state = 'Tracked'
        
        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter()
        cx, cy = self._get_center()
        self.kalman_filter.x[:2] = [cx, cy]
        
        self.is_activated = False
        self.track_len = 0
    
    def _get_center(self):
        """Get bbox center"""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def predict(self):
        """Predict next position"""
        predicted_pos = self.kalman_filter.predict()
        # Update bbox center with prediction
        cx, cy = predicted_pos
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        self.bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    
    def update(self, bbox, confidence, frame_id):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update Kalman filter
        cx, cy = self._get_center()
        self.kalman_filter.update([cx, cy])
        
        self.state = 'Tracked'
    
    def activate(self, frame_id):
        """Activate track"""
        self.track_id = STrack.track_id
        STrack.track_id += 1
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


def iou(bbox1, bbox2):
    """Calculate IoU between two bboxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class BYTETracker:
    """ByteTrack implementation for soccer ball tracking"""
    
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.8, high_thresh=0.6, low_thresh=0.1):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        
        self.frame_id = 0
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Main tracking update function"""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Separate high and low confidence detections
        high_dets = [det for det in detections if det.confidence >= self.high_thresh]
        low_dets = [det for det in detections if det.confidence >= self.low_thresh and det.confidence < self.high_thresh]
        
        # Create STrack objects
        high_stracks = [STrack(det.bbox, det.confidence, self.frame_id) for det in high_dets]
        
        # Predict existing tracks
        for track in self.tracked_stracks:
            track.predict()
        
        # First association with high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_stracks, high_stracks, self.match_thresh)
        
        # Update matched tracks
        for m in matched:
            track_idx, det_idx = m
            track = self.tracked_stracks[track_idx]
            det = high_stracks[det_idx]
            track.update(det.bbox, det.confidence, self.frame_id)
            activated_starcks.append(track)
        
        # Second association with low confidence detections
        low_stracks = [STrack(det.bbox, det.confidence, self.frame_id) for det in low_dets]
        unmatched_track_stracks = [self.tracked_stracks[i] for i in unmatched_tracks]
        
        matched_low, unmatched_tracks_low, unmatched_dets_low = self._associate(
            unmatched_track_stracks, low_stracks, 0.5)  # Lower threshold
        
        for m in matched_low:
            track_idx, det_idx = m
            track = unmatched_track_stracks[track_idx]
            det = low_stracks[det_idx]
            track.update(det.bbox, det.confidence, self.frame_id)
            activated_starcks.append(track)
        
        # Handle unmatched tracks
        for track in unmatched_track_stracks:
            if track not in activated_starcks:
                track.state = 'Lost'
                lost_stracks.append(track)
        
        # Initialize new tracks from unmatched high confidence detections
        unmatched_high_dets = [high_stracks[i] for i in unmatched_dets]
        for det in unmatched_high_dets:
            if det.confidence >= self.track_thresh:
                det.activate(self.frame_id)
                activated_starcks.append(det)
        
        # SOCCER OPTIMIZATION: Also initialize tracks from unmatched low confidence
        # if we have very few active tracks (soccer ball might be temporarily occluded)
        if len(activated_starcks) < 2:  # If we have less than 2 active tracks
            unmatched_low_high_dets = [low_stracks[i] for i in unmatched_dets_low]
            for det in unmatched_low_high_dets[:1]:  # Take best unmatched low confidence
                if det.confidence >= 0.15:  # Minimum threshold
                    det.activate(self.frame_id)
                    activated_starcks.append(det)
        
        # Update track lists
        self.tracked_stracks = activated_starcks
        
        # Remove old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks if self.frame_id - t.frame_id <= self.max_time_lost]
        
        # Convert to output format
        output_tracks = []
        for track in self.tracked_stracks:
            output_tracks.append(Track(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                state=track.state,
                frame_id=self.frame_id,
                start_frame=track.start_frame
            ))
        
        return output_tracks
    
    def _associate(self, tracks, detections, thresh):
        """Associate tracks with detections using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i][j] = iou(track.bbox, det.bbox)
        
        # Hungarian assignment (simplified greedy approach)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Greedy matching
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find maximum IoU
            max_iou = 0
            max_i, max_j = -1, -1
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i][j] > max_iou:
                        max_iou = iou_matrix[i][j]
                        max_i, max_j = i, j
            
            if max_iou >= thresh:
                matched.append([max_i, max_j])
                unmatched_tracks.remove(max_i)
                unmatched_dets.remove(max_j)
            else:
                break
        
        return matched, unmatched_tracks, unmatched_dets