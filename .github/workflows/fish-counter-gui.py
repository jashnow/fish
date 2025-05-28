import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QHBoxLayout, QGroupBox, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2
from ultralytics import YOLO
import os

# --- GRAFİK EKİ ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FishBarChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 2), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fish_counts = []

    def update_chart(self, count):
        self.fish_counts.append(count)
        self.ax.clear()
        self.ax.bar(range(len(self.fish_counts)), self.fish_counts, color='#039be5')
        self.ax.set_title("Fish Count (Frame Based)", fontsize=10)
        self.ax.set_xlabel("Frame", fontsize=8)
        self.ax.set_ylabel("Fish Count", fontsize=8)
        self.draw()

# --- ANA UYGULAMA ---
class FishCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fish Detection and Analysis")
        self.setGeometry(100, 100, 1300, 750)
        self.setStyleSheet("background-color: #e0f7fa;")

        # GÜNCELLENMİŞ MODEL YOLU
        model_path = r"C:\Users\JashNow\Desktop\sinan\eğitilen model ve projenin ana kodları\datasets\runs\detect\train3\weights\best.pt"
        self.model = YOLO(model_path)

        self.cap = None

        self.image_label = QLabel("Loading...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #0288d1; background-color: #b3e5fc;")

        self.count_label = QLabel("Fish Count: 0")
        self.count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.count_label.setStyleSheet("color: #01579b;")

        self.param_display = QTextEdit()
        self.param_display.setReadOnly(True)
        self.param_display.setFixedHeight(150)
        self.param_display.setStyleSheet("background-color: #ffffff; border: 1px solid #0288d1;")

        self.coord_display = QTextEdit()
        self.coord_display.setReadOnly(True)
        self.coord_display.setFixedHeight(150)
        self.coord_display.setStyleSheet("background-color: #ffffff; border: 1px solid #0288d1;")

        self.select_button = QPushButton("Select media")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        for btn in [self.select_button, self.start_button, self.stop_button]:
            btn.setStyleSheet("background-color: #4fc3f7; font-weight: bold; color: white; padding: 5px;")

        self.select_button.clicked.connect(self.select_file)
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.file_path = None
        self.is_image = False

        self.chart = FishBarChart(self)  # GRAFİK BİLEŞENİ

        self.init_ui()

    def init_ui(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.count_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        info_group = QGroupBox("Test Parameters")
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.param_display)
        info_group.setLayout(info_layout)

        coord_group = QGroupBox("Fish Coordinates")
        coord_layout = QVBoxLayout()
        coord_layout.addWidget(self.coord_display)
        coord_group.setLayout(coord_layout)

        right_layout = QVBoxLayout()
        right_layout.addLayout(button_layout)
        right_layout.addWidget(info_group)
        right_layout.addWidget(coord_group)
        right_layout.addWidget(self.chart)  # GRAFİĞİ EKLE

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def select_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Media Files (*.mp4 *.avi *.jpg *.png)")
        if self.file_path:
            ext = os.path.splitext(self.file_path)[-1].lower()
            self.is_image = ext in ['.jpg', '.png']
            self.chart.fish_counts.clear()  # GRAFİĞİ SIFIRLA
            if self.is_image:
                self.display_image(self.file_path)
            else:
                self.cap = cv2.VideoCapture(self.file_path)
                self.param_display.append(f"Select Video: {self.file_path}")

    def start_video(self):
        if self.cap and not self.timer.isActive():
            self.timer.start(30)
            self.param_display.append("The video started playing.")

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            self.param_display.append("The video was paused and closed.")

    def display_image(self, path):
        frame = cv2.imread(path)
        self.process_frame(frame)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return
            self.process_frame(frame)

    def process_frame(self, frame):
        results = self.model(frame)[0]
        fish_count = len(results.boxes)
        self.count_label.setText(f"Fish Count: {fish_count}")

        self.param_display.append(
            f"Frame analysis: Fish Count = {fish_count}, Size = {frame.shape[1]}x{frame.shape[0]}"
        )

        self.coord_display.clear()
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            self.coord_display.append(f"Coordinate: ({center_x}, {center_y})")

        annotated_frame = results.plot()
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

        self.chart.update_chart(fish_count)  # GRAFİĞİ GÜNCELLE

    def closeEvent(self, event):
        self.stop_video()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FishCounterApp()
    window.show()
    sys.exit(app.exec_())
