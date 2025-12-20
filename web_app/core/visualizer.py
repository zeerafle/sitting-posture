import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


class PostureVisualizer:
    """Handles visualization of pose and analytics."""

    def __init__(self):
        self.skeleton_connections = [
            (1, 2),  # shoulders
            (1, 3), (3, 5),  # left arm
            (2, 4), (4, 6),  # right arm
            (1, 7), (2, 8),  # torso
            (7, 8),  # hips
            (7, 9), (9, 11),  # left leg
            (8, 10), (10, 12)  # right leg
        ]

    def draw_keypoints(self, frame, keypoints, prediction_label=None, confidence=None):
        """Draw keypoints and skeleton on the frame."""
        if frame is None or keypoints is None:
            return frame

        # Make a copy to avoid modifying original
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        # Draw connections
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]

                if start_point[2] > 0.3 and end_point[2] > 0.3:
                    start_pos = (int(start_point[1] * w), int(start_point[0] * h))
                    end_pos = (int(end_point[1] * w), int(end_point[0] * h))
                    cv2.line(annotated_frame, start_pos, end_pos, (255, 255, 255), 2)

        # Draw keypoints
        for i, (y, x, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(annotated_frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
                # Optional: add keypoint numbers
                # cv2.putText(annotated_frame, str(i), (int(x * w) + 5, int(y * h) - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Add prediction text if provided
        if prediction_label is not None:
            color = (0, 255, 0) if "Ergonomic" in prediction_label else (255, 0, 0)
            cv2.putText(annotated_frame, f"{prediction_label}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if confidence is not None:
                cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated_frame

    def create_analytics_plot(self, prediction_history, timestamps):
        """Generate analytics plot showing posture over time."""
        if len(prediction_history) < 2:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'No data available yet',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Posture Analytics')
            plt.tight_layout()
            return fig

        # Create time series plot
        fig, ax = plt.subplots(figsize=(10, 4))

        # Convert to lists
        times = list(timestamps)
        predictions = list(prediction_history)

        # Create scatter plot
        colors = ['green' if p == 0 else 'red' for p in predictions]
        ax.scatter(times, predictions, c=colors, alpha=0.6, s=50)

        # Add trend line (moving average)
        if len(predictions) > 10:
            window_size = min(20, len(predictions) // 2)
            moving_avg = []
            for i in range(len(predictions)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(predictions), i + window_size // 2)
                avg = np.mean(predictions[start_idx:end_idx])
                moving_avg.append(avg)

            ax.plot(times, moving_avg, 'b-', alpha=0.7, linewidth=2, label='Trend')
            ax.legend()

        # Formatting
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Good Posture', 'Poor Posture'])
        ax.set_title('Posture Analysis Over Time')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        if len(times) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def generate_statistics(self, prediction_history, timestamps):
        """Generate session statistics."""
        if len(prediction_history) == 0:
            return "No data collected yet"

        total_frames = len(prediction_history)
        good_posture_frames = sum(1 for p in prediction_history if p == 0)
        poor_posture_frames = total_frames - good_posture_frames

        good_percentage = (good_posture_frames / total_frames) * 100
        poor_percentage = (poor_posture_frames / total_frames) * 100

        # Calculate session duration
        if len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "0:00:00"

        stats = f"""
        📊 **Session Statistics**

        **Session Duration:** {duration_str}
        **Total Frames Analyzed:** {total_frames}

        **Posture Breakdown:**
        - ✅ Good Posture: {good_posture_frames} frames ({good_percentage:.1f}%)
        - ❌ Poor Posture: {poor_posture_frames} frames ({poor_percentage:.1f}%)

        **Recommendations:**
        {"🎉 Great job! Keep maintaining good posture." if good_percentage >= 70 else "⚠️ Try to improve posture. Take breaks and adjust your setup."}
        """

        return stats
