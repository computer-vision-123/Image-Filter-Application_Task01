import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from tab_noise_filters import NoiseTab
from tab_edge_freq import EdgeTab
from tab_hist_contrast import HistogramContrastTab
from tab_color_hybrid import FrequencyFilterTab, ColorHybridTab
from Helpers.styles import load_app_font


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Suite")
        self.resize(700, 700)

        self.setStyleSheet("""
            * {
                font-family: "Poppins", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
            }
            QMainWindow { background-color: #f5f5f5; }
            QTabWidget::pane {
                border: 2px solid #87ceeb;
                border-radius: 8px;
                background-color: white;
                margin: 2px;
            }
            QTabBar::tab {
                background-color: #bbbbbb;
                color: #2c3e50;
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
            }
            QTabBar::tab:selected  { background-color: #87ceeb; color: white; }
            QTabBar::tab:hover:!selected { background-color: #98d8c8; color: white; }
        """)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(NoiseTab(),             "1. Noise & Filters")
        self.tabs.addTab(EdgeTab(),              "2. Edge Detection")
        self.tabs.addTab(HistogramContrastTab(), "3. Histogram & Contrast")
        self.tabs.addTab(ColorHybridTab(),       "4. Color & Hybrid")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_app_font(app)   # registers Poppins and sets it as the app-wide font
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())