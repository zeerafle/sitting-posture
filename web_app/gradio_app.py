import gradio as gr
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core.pose_analyzer import PostureAnalyzer
from core.visualizer import PostureVisualizer


class GradioPostureApp:
    """Main Gradio application class."""

    def __init__(self):
        """Initialize the Gradio application."""
        # Set project root (parent directory of web_app)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Initialize components
        self.analyzer = PostureAnalyzer(model_type="xgb", project_root=self.project_root)
        self.visualizer = PostureVisualizer()

        print("Gradio Posture App initialized successfully!")

    def process_frame(self, frame, model_type):
        """Process a frame from the webcam stream."""
        # Update model if changed
        if self.analyzer.model_type != model_type:
            status = self.analyzer.update_model_type(model_type)
            print(status)

        if frame is None:
            return frame, "No frame", 0.0, "⚠️ No Input", "0 ms"

        # Process frame through analyzer
        keypoints, prediction_label, confidence, status_emoji, processing_time = self.analyzer.process_frame(frame)

        # Format processing time
        processing_time_str = f"{processing_time:.1f} ms"

        # Visualize results
        if keypoints is not None and prediction_label != "No Pose Detected":
            annotated_frame = self.visualizer.draw_keypoints(
                frame, keypoints, prediction_label, confidence
            )
        else:
            annotated_frame = frame.copy()
            # Add status text for no pose detected
            if prediction_label == "No Pose Detected":
                import cv2
                cv2.putText(annotated_frame, "No pose detected",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated_frame, prediction_label, confidence, status_emoji, processing_time_str

    def update_analytics(self):
        """Update analytics plot and statistics."""
        plot = self.visualizer.create_analytics_plot(
            self.analyzer.prediction_history,
            self.analyzer.timestamps
        )
        stats = self.visualizer.generate_statistics(
            self.analyzer.prediction_history,
            self.analyzer.timestamps
        )
        return plot, stats

    def clear_data(self):
        """Clear analytics data."""
        self.analyzer.clear_history()
        return self.update_analytics()

    def create_interface(self):
        """Create and return the Gradio interface."""
        with gr.Blocks(title="Posture Analysis System", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🪑 Real-Time Posture Analysis System")
            gr.Markdown("Monitor your sitting posture in real-time using AI-powered pose detection.")

            with gr.Row():
                with gr.Column(scale=2):
                    # Main video interface
                    with gr.Group():
                        gr.Markdown("### 📹 Live Camera Feed")

                        # Model settings
                        model_type = gr.Dropdown(
                            choices=["xgb", "adaboost", "nn"],
                            value="nn",
                            label="Model Type",
                            info="NN is recommended for best performance"
                        )

                        # Webcam input
                        webcam = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                            label="Camera Input"
                        )

                with gr.Column(scale=1):
                    # Live status
                    with gr.Group():
                        gr.Markdown("### 📊 Live Status")

                        posture_status = gr.Textbox(
                            label="Current Posture",
                            value="Ready to analyze",
                            interactive=False
                        )

                        confidence_score = gr.Number(
                            label="Confidence Score",
                            value=0.0,
                            interactive=False
                        )

                        status_indicator = gr.Textbox(
                            label="Status",
                            value="🟡 Ready",
                            interactive=False
                        )

                        # Add processing time display
                        processing_time = gr.Textbox(
                            label="Processing Time",
                            value="0 ms",
                            interactive=False
                        )

            # Analytics section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 Analytics Dashboard")

                    analytics_plot = gr.Plot(label="Posture Over Time")

                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Data", variant="stop")

                    session_stats = gr.Markdown("Click 'Refresh Analytics' to see statistics")

            # Auto-refresh timer (invisible component that triggers updates)
            auto_refresh_timer = gr.Timer(value=30)  # Refresh every 30 seconds

            # Set up streaming
            webcam.stream(
                fn=self.process_frame,
                inputs=[webcam, model_type],
                outputs=[webcam, posture_status, confidence_score, status_indicator, processing_time],
                stream_every=0.1,
                time_limit=3600  # 1 hour limit
            )

            # Event handlers
            refresh_btn.click(
                fn=self.update_analytics,
                outputs=[analytics_plot, session_stats]
            )

            clear_btn.click(
                fn=self.clear_data,
                outputs=[analytics_plot, session_stats]
            )

            # Auto-refresh analytics using timer
            auto_refresh_timer.tick(
                fn=self.update_analytics,
                outputs=[analytics_plot, session_stats]
            )

            # Initial load of analytics
            app.load(
                fn=self.update_analytics,
                outputs=[analytics_plot, session_stats]
            )

            # Instructions
            with gr.Accordion("📋 Instructions & Tips", open=False):
                gr.Markdown("""
                ### How to Use:
                1. **Allow camera access** when prompted by your browser
                2. **Position yourself** so your upper body is visible in the camera
                3. **Select your preferred model** (XGBoost is recommended)
                4. **Monitor your posture** in real-time with immediate feedback

                ### Tips for Best Results:
                - 💡 **Good lighting**: Ensure your workspace is well-lit
                - 👤 **Full visibility**: Keep your shoulders, arms, and head visible
                - 📷 **Stable camera**: Use a stable camera position at eye level
                - 🪑 **Natural posture**: Sit naturally and avoid covering key body parts
                - 🔄 **Regular breaks**: Use the analytics to track and improve over time

                ### Understanding the Results:
                - 🟢 **Green overlay**: Good ergonomic posture detected
                - 🔴 **Red overlay**: Poor posture detected - adjust your position
                - ⚪ **White lines**: Pose skeleton showing detected body joints
                - 📊 **Analytics**: Track your posture trends over time

                ### Model Information:
                - **XGBoost**: Best overall performance and speed (recommended)
                - **AdaBoost**: Good accuracy with ensemble learning
                - **Neural Network**: Deep learning approach with high precision
                """)

        return app


def main():
    """Main function to run the Gradio app."""
    # Create the application
    app_instance = GradioPostureApp()
    gradio_app = app_instance.create_interface()

    # Launch the application
    gradio_app.launch(
        # share=True,  # Create public link for sharing
        # server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
