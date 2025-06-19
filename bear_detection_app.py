import sys
import os
import cv2
import numpy as np
from PIL import Image # Used for saving QImage to disk for temporary files

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStackedWidget,
    QScrollArea, QMessageBox, QSizePolicy, QProgressBar,
    QSlider, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# YOLO model import
# Ensure 'ultralytics' is installed: pip install ultralytics
from ultralytics import YOLO

# For creating temporary directories for YOLO output
import tempfile
import shutil # For cleaning up temporary directories

# --- 1. Prediction Worker Thread ---
# This class runs the YOLO prediction in a separate thread
# to prevent the UI from freezing during processing.
class PredictionWorker(QThread):
    # Signals to communicate back to the main UI thread
    finished = pyqtSignal(list, str) # Emits list of paths to processed files and output dir
    error = pyqtSignal(str) # Emits error message
    progress = pyqtSignal(int) # Emits current progress (0-100)

    def __init__(self, model_path, input_files):
        super().__init__()
        self.model_path = model_path
        self.input_files = input_files
        self.model = None
        self.temp_output_dir = None # To store YOLO's output

    def run(self):
        try:
            # Load the YOLO model
            self.model = YOLO(self.model_path)

            # Create a unique temporary directory for this prediction run's output
            self.temp_output_dir = tempfile.mkdtemp(prefix="yolo_output_")
            print(f"YOLO output will be saved to: {self.temp_output_dir}")

            processed_file_paths = []
            total_files = len(self.input_files)

            for i, file_path in enumerate(self.input_files):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                    # --- Video: process frame by frame ---
                    cap = cv2.VideoCapture(file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_idx = 0
                    video_save_dir = os.path.join(self.temp_output_dir, f"video_{i}")
                    os.makedirs(video_save_dir, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = os.path.join(video_save_dir, os.path.basename(file_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Run detection on frame
                        results = self.model(frame)
                        if results and len(results) > 0:
                            frame_with_boxes = results[0].plot()
                            out.write(frame_with_boxes)
                        else:
                            out.write(frame)
                        frame_idx += 1
                        progress = int((frame_idx / total_frames) * 100)
                        self.progress.emit(progress)
                    cap.release()
                    out.release()
                    processed_file_paths.append(out_path)
                    self.progress.emit(100)  # Ensure 100% at end
                else:
                    # --- Image: original logic ---
                    self.progress.emit(int((i / total_files) * 100))
                    results = self.model(
                        file_path,
                        save=True,
                        project=self.temp_output_dir,
                        name='prediction_run'
                    )
                    if results and len(results) > 0:
                        yolo_save_dir = results[0].save_dir
                        original_filename = os.path.basename(file_path)
                        processed_path = os.path.join(yolo_save_dir, original_filename)
                        if os.path.exists(processed_path):
                            processed_file_paths.append(processed_path)
                        else:
                            print(f"Warning: Processed file not found at {processed_path}")
                            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_with_boxes_np = results[0].plot()
                                img_with_boxes_rgb = cv2.cvtColor(img_with_boxes_np, cv2.COLOR_BGR2RGB)
                                h, w, ch = img_with_boxes_rgb.shape
                                bytes_per_line = ch * w
                                qImg = QImage(img_with_boxes_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                                temp_img_path = os.path.join(self.temp_output_dir, f"annotated_{original_filename}")
                                qImg.save(temp_img_path)
                                processed_file_paths.append(temp_img_path)
                            else:
                                self.error.emit(f"Could not locate or display processed file for: {original_filename}")
                                return
                    else:
                        self.error.emit(f"YOLO did not return results for {file_path}")
                        return
            self.progress.emit(100)
            self.finished.emit(processed_file_paths, self.temp_output_dir)
        except Exception as e:
            if self.temp_output_dir and os.path.exists(self.temp_output_dir):
                shutil.rmtree(self.temp_output_dir)
            self.error.emit(f"Error during prediction: {e}")

# --- 2. Main Application Window ---
class BearDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bear Detection Desktop Application")
        self.setGeometry(100, 100, 1400, 900) # Increased window size for video support

        self.input_files = [] # Stores paths of uploaded files
        self.prediction_thread = None
        self.current_temp_output_dir = None # To keep track of the current temp dir for cleanup
        self.current_image_index = 0
        self.current_video_index = 0
        self.image_paths = []
        self.video_paths = []
        
        # Video player components
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self) # Main horizontal layout for left and right panels

        # --- Left Panel (Input/Controls) ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setStyleSheet(
            "background-color: #4A90E2; border-radius: 15px; padding: 20px; color: white;"
        )
        left_panel_widget.setFixedWidth(450)

        title_label = QLabel("Bear Detection Application")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(title_label)

        lightbulb_label = QLabel("💡")
        lightbulb_label.setStyleSheet("font-size: 80px; margin-bottom: 30px;")
        lightbulb_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(lightbulb_label)

        upload_text_label = QLabel("Upload video or images")
        upload_text_label.setStyleSheet("font-size: 18px; margin-bottom: 10px;")
        upload_text_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(upload_text_label)

        self.file_path_display = QLabel("No files selected")
        self.file_path_display.setStyleSheet(
            "background-color: #E0E0E0; color: #333; padding: 10px; border-radius: 8px; font-size: 14px;"
        )
        self.file_path_display.setAlignment(Qt.AlignCenter)
        self.file_path_display.setWordWrap(True)
        left_panel_layout.addWidget(self.file_path_display)

        upload_button = QPushButton("Upload Files")
        upload_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2ECC71;
                color: white;
                padding: 12px 25px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #27AE60;
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            QPushButton:pressed {
                background-color: #229954;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
                color: #555;
                box-shadow: none;
            }
            """
        )
        upload_button.clicked.connect(self.upload_files)
        left_panel_layout.addWidget(upload_button)

        # Detect button
        self.predict_button = QPushButton("Detect")
        self.predict_button.setStyleSheet(
            """
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 12px 25px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #C0392B;
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            QPushButton:pressed {
                background-color: #922B21;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
                color: #555;
                box-shadow: none;
            }
            """
        )
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setVisible(False)
        left_panel_layout.addWidget(self.predict_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #27AE60;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #90EE90; /* Light green */
                width: 20px;
            }
            """
        )
        self.progress_bar.setVisible(False)
        left_panel_layout.addWidget(self.progress_bar)

        left_panel_layout.addStretch(1)

        main_layout.addWidget(left_panel_widget)

        # --- Right Panel (Display Area) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setStyleSheet(
            "background-color: #F8F8F8; border-radius: 15px; padding: 20px;"
        )

        # Tab widget for Images and Videos
        self.media_tabs = QTabWidget()
        self.media_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498DB;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #5DADE2;
                color: white;
            }
            """
        )

        # Images Tab
        self.setup_images_tab()
        
        # Videos Tab
        self.setup_videos_tab()

        # Placeholder
        self.placeholder_label = QLabel("Upload images or videos to see them here.")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 18px; color: #666; font-style: italic;")

        self.main_display_stack = QStackedWidget()
        self.main_display_stack.addWidget(self.placeholder_label)  # Index 0
        self.main_display_stack.addWidget(self.media_tabs)  # Index 1

        right_panel_layout.addWidget(self.main_display_stack)
        main_layout.addWidget(right_panel_widget, 1)

    def setup_images_tab(self):
        images_widget = QWidget()
        images_layout = QVBoxLayout(images_widget)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ddd; border-radius: 10px; background-color: white;")
        self.image_label.setMinimumSize(700, 500)
        images_layout.addWidget(self.image_label)
        
        # Image navigation controls
        img_nav_container = QWidget()
        img_nav_layout = QHBoxLayout(img_nav_container)
        img_nav_layout.setAlignment(Qt.AlignCenter)
        
        self.prev_image_button = QPushButton('← Previous')
        self.prev_image_button.setStyleSheet(self.get_nav_button_style())
        self.prev_image_button.clicked.connect(self.show_prev_image)

        self.image_counter_label = QLabel("0 / 0")
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        self.image_counter_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50; margin: 0 20px;")

        self.next_image_button = QPushButton('Next →')
        self.next_image_button.setStyleSheet(self.get_nav_button_style())
        self.next_image_button.clicked.connect(self.show_next_image)

        # Download button for images
        self.download_image_button = QPushButton('Download Image')
        self.download_image_button.setStyleSheet(self.get_nav_button_style())
        self.download_image_button.clicked.connect(self.download_current_image)

        img_nav_layout.addWidget(self.prev_image_button)
        img_nav_layout.addWidget(self.image_counter_label)
        img_nav_layout.addWidget(self.next_image_button)
        img_nav_layout.addWidget(self.download_image_button)

        images_layout.addWidget(img_nav_container)
        
        self.media_tabs.addTab(images_widget, "Images")

    def setup_videos_tab(self):
        videos_widget = QWidget()
        videos_layout = QVBoxLayout(videos_widget)
        
        # Video display area
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(700, 500)
        self.video_widget.setStyleSheet("border: 2px solid #ddd; border-radius: 10px; background-color: black;")
        self.media_player.setVideoOutput(self.video_widget)
        videos_layout.addWidget(self.video_widget)
        
        # Video controls
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        
        # Play controls
        play_controls = QWidget()
        play_layout = QHBoxLayout(play_controls)
        play_layout.setAlignment(Qt.AlignCenter)
        
        self.play_button = QPushButton('Play')
        self.play_button.setStyleSheet(self.get_control_button_style())
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.stop_button = QPushButton('Stop')
        self.stop_button.setStyleSheet(self.get_control_button_style())
        self.stop_button.clicked.connect(self.stop_video)
        
        play_layout.addWidget(self.play_button)
        play_layout.addWidget(self.stop_button)
        controls_layout.addWidget(play_controls)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                border: 1px solid #2980B9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498DB;
                border-radius: 4px;
            }
            """
        )
        controls_layout.addWidget(self.position_slider)
        
        # Video navigation
        video_nav_container = QWidget()
        video_nav_layout = QHBoxLayout(video_nav_container)
        video_nav_layout.setAlignment(Qt.AlignCenter)

        self.prev_video_button = QPushButton('← Previous Video')
        self.prev_video_button.setStyleSheet(self.get_nav_button_style())
        self.prev_video_button.clicked.connect(self.show_prev_video)

        self.video_counter_label = QLabel("0 / 0")
        self.video_counter_label.setAlignment(Qt.AlignCenter)
        self.video_counter_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50; margin: 0 20px;")

        self.next_video_button = QPushButton('Next Video →')
        self.next_video_button.setStyleSheet(self.get_nav_button_style())
        self.next_video_button.clicked.connect(self.show_next_video)

        # Download button for videos
        self.download_video_button = QPushButton('Download Video')
        self.download_video_button.setStyleSheet(self.get_nav_button_style())
        self.download_video_button.clicked.connect(self.download_current_video)

        video_nav_layout.addWidget(self.prev_video_button)
        video_nav_layout.addWidget(self.video_counter_label)
        video_nav_layout.addWidget(self.next_video_button)
        video_nav_layout.addWidget(self.download_video_button)

        controls_layout.addWidget(video_nav_container)
        videos_layout.addWidget(controls_container)
        
        # Connect media player signals
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        
        self.media_tabs.addTab(videos_widget, "Videos")

    def get_nav_button_style(self):
        return """
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """

    def get_control_button_style(self):
        return """
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
        """

    def upload_files(self):
        options = QFileDialog.Options()
        file_filters = (
            "Media Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.wmv *.flv);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;"
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;"
            "All Files (*)"
        )
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images or Videos", "", file_filters, options=options
        )

        if file_paths:
            self.input_files = file_paths
            display_text = f"{len(file_paths)} file(s) selected:\n" + "\n".join([os.path.basename(f) for f in file_paths[:10]])
            if len(file_paths) > 10:
                display_text += "\n..."
            self.file_path_display.setText(display_text)
            self.predict_button.setVisible(True)
            self.separate_media_files(self.input_files)
            self.main_display_stack.setCurrentIndex(1)  # Show media tabs
            self.cleanup_temp_dir()

    def separate_media_files(self, file_paths):
        # Separate images and videos
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        
        self.image_paths = [f for f in file_paths if os.path.splitext(f)[1].lower() in image_extensions]
        self.video_paths = [f for f in file_paths if os.path.splitext(f)[1].lower() in video_extensions]
        
        self.current_image_index = 0
        self.current_video_index = 0
        
        # Update displays
        self.update_image_display()
        self.update_video_display()
        
        # Set appropriate tab
        if self.image_paths and not self.video_paths:
            self.media_tabs.setCurrentIndex(0)  # Images tab
        elif self.video_paths and not self.image_paths:
            self.media_tabs.setCurrentIndex(1)  # Videos tab
        elif self.image_paths and self.video_paths:
            self.media_tabs.setCurrentIndex(0)  # Default to images tab

    def update_image_display(self):
        if not self.image_paths:
            self.image_label.setText("No images to display")
            self.image_counter_label.setText("0 / 0")
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            return
        
        current_path = self.image_paths[self.current_image_index]
        pixmap = QPixmap(current_path)
        
        if not pixmap.isNull():
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size.width() - 20, 
                label_size.height() - 20, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText(f"Could not load image: {os.path.basename(current_path)}")
        
        self.image_counter_label.setText(f"{self.current_image_index + 1} / {len(self.image_paths)}")
        self.prev_image_button.setEnabled(self.current_image_index > 0)
        self.next_image_button.setEnabled(self.current_image_index < len(self.image_paths) - 1)

    def update_video_display(self):
        if not self.video_paths:
            self.video_counter_label.setText("0 / 0")
            self.prev_video_button.setEnabled(False)
            self.next_video_button.setEnabled(False)
            return
        
        current_path = self.video_paths[self.current_video_index]
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(current_path)))
        
        self.video_counter_label.setText(f"{self.current_video_index + 1} / {len(self.video_paths)}")
        self.prev_video_button.setEnabled(self.current_video_index > 0)
        self.next_video_button.setEnabled(self.current_video_index < len(self.video_paths) - 1)

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()

    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.update_image_display()

    def show_prev_video(self):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.update_video_display()

    def show_next_video(self):
        if self.current_video_index < len(self.video_paths) - 1:
            self.current_video_index += 1
            self.update_video_display()

    def toggle_playback(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_video(self):
        self.media_player.stop()

    def set_position(self, position):
        self.media_player.setPosition(position)

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText('Pause')
            self.position_timer.start(100)
        else:
            self.play_button.setText('Play')
            self.position_timer.stop()

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def update_position(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.position_slider.setValue(self.media_player.position())

    def resizeEvent(self, event):
        if hasattr(self, 'image_paths') and self.image_paths and self.main_display_stack.currentIndex() == 1:
            self.update_image_display()
        super().resizeEvent(event)

    def run_prediction(self):
        if not self.input_files:
            QMessageBox.warning(self, "No Files Selected", "Please upload images or videos first.")
            return

        self.cleanup_temp_dir()
        self.predict_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        model_path = "trenirani modeli/best.pt"
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Model Not Found",
                                 f"YOLO model not found at: {model_path}\nPlease ensure the model path is correct and the model file exists.")
            self.predict_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return

        self.prediction_thread = PredictionWorker(model_path, self.input_files)
        self.prediction_thread.finished.connect(self.prediction_finished)
        self.prediction_thread.error.connect(self.prediction_error)
        self.prediction_thread.progress.connect(self.progress_bar.setValue)
        self.prediction_thread.start()

    def prediction_finished(self, processed_file_paths, temp_output_dir):
        self.predict_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_temp_output_dir = temp_output_dir
        self.separate_media_files(processed_file_paths)
        QMessageBox.information(self, "Prediction Complete", "Object detection finished successfully!")

    def prediction_error(self, message):
        self.predict_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cleanup_temp_dir()
        QMessageBox.critical(self, "Prediction Error", f"An error occurred: {message}")

    def cleanup_temp_dir(self):
        if self.current_temp_output_dir and os.path.exists(self.current_temp_output_dir):
            try:
                print(f"Cleaning up temporary directory: {self.current_temp_output_dir}")
                shutil.rmtree(self.current_temp_output_dir)
                self.current_temp_output_dir = None
            except Exception as e:
                print(f"Error cleaning up temp directory {self.current_temp_output_dir}: {e}")

    def closeEvent(self, event):
        self.cleanup_temp_dir()
        self.media_player.stop()
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.prediction_thread.quit()
            self.prediction_thread.wait()
        event.accept()

    def download_current_image(self):
        if not self.image_paths:
            return
        current_path = self.image_paths[self.current_image_index]
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.basename(current_path), "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if save_path:
            try:
                shutil.copy(current_path, save_path)
                QMessageBox.information(self, "Download Complete", f"Image saved to: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Download Error", f"Failed to save image: {e}")

    def download_current_video(self):
        if not self.video_paths:
            return
        current_path = self.video_paths[self.current_video_index]
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", os.path.basename(current_path), "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files (*)", options=options)
        if save_path:
            try:
                shutil.copy(current_path, save_path)
                QMessageBox.information(self, "Download Complete", f"Video saved to: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Download Error", f"Failed to save video: {e}")
# --- 3. Main Execution Block ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    default_font = QApplication.font()
    default_font.setPointSize(10)
    app.setFont(default_font)

    window = BearDetectionApp()
    window.show()
    sys.exit(app.exec_())