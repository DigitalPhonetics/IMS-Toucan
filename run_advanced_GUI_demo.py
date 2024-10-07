import sys

import numpy as np
import pyqtgraph as pg
import scipy.io.wavfile
import sounddevice
import torch.cuda
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QCursor
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from huggingface_hub import hf_hub_download

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.utils import load_json_from_path


class DraggableScatter(pg.ScatterPlotItem):
    pointMoved = pyqtSignal(int, float)  # Emits index and new y-value

    def __init__(self, x, y, pen=None, brush=None, size=10, **kwargs):
        super().__init__(x=x, y=y, pen=pen, brush=brush, size=size, **kwargs)
        self.setAcceptHoverEvents(True)
        self.dragging = False
        self.selected_point = None
        self.x = list(x)
        self.y = list(y)

    def getViewBox(self):
        """
        Traverse up the parent hierarchy to locate the ViewBox.
        Returns the ViewBox if found, else None.
        """
        parent = self.parentItem()
        while parent is not None:
            if isinstance(parent, pg.ViewBox):
                return parent
            parent = parent.parentItem()
        return None

    def mousePressEvent(self, event):
        threshold = 100
        if event.button() == Qt.LeftButton:
            vb = self.getViewBox()
            if vb is None:
                super().mousePressEvent(event)
                return
            mousePoint = vb.mapSceneToView(event.scenePos())
            x_click = mousePoint.x()
            # Find the closest point
            min_dist = float('inf')
            closest_idx = None
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                dist = abs(x - x_click)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            if min_dist < threshold:
                self.selected_point = closest_idx
                self.dragging = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point is not None:
            vb = self.getViewBox()
            if vb is None:
                super().mouseMoveEvent(event)
                return
            mousePoint = vb.mapSceneToView(event.scenePos())
            new_y = mousePoint.y()
            if 0 < new_y < 2:
                self.y[self.selected_point] = new_y
                self.setData(x=self.x, y=self.y)
                self.pointMoved.emit(self.selected_point, new_y)
                event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_point = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverEvent(self, event):
        threshold = 100
        if event.isExit():
            self.setCursor(Qt.ArrowCursor)
        else:
            vb = self.getViewBox()
            if vb is None:
                self.setCursor(Qt.ArrowCursor)
                return
            mousePoint = vb.mapSceneToView(event.scenePos())
            x_hover = mousePoint.x()
            # Check if hovering near a point
            min_dist = float('inf')
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                dist = abs(x - x_hover)
                if dist < min_dist:
                    min_dist = dist
            if min_dist < threshold:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))


class TTSInterface(QMainWindow):
    def __init__(self, tts_interface: ToucanTTSInterface):
        super().__init__()

        path_to_iso_list = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="iso_to_fullname.json")
        iso_to_name = load_json_from_path(path_to_iso_list)
        self.name_to_iso = dict()
        for iso in iso_to_name:
            self.name_to_iso[iso_to_name[iso]] = iso
        text_selection = [iso_to_name[iso_code] for iso_code in iso_to_name]
        self.tts_backend = tts_interface

        # Define Placeholders
        self.word_boundaries = []
        self.pitch_curve = None
        self.phonemes = []
        self.durations = None
        self.pitch = None
        self.audio_file_path = None
        self.result_audio = None
        self.min_duration = 1
        self.slider_val = 100
        self.durations_are_scaled = False
        self.prev_slider_val_for_denorm = 100

        self.setWindowTitle("TTS Model Interface")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.main_layout.setSpacing(15)  # spacing between widgets
        self.main_layout.setContentsMargins(20, 50, 20, 30)  # Left, Top, Right, Bottom

        # Add Text Input
        self.text_input_layout = QHBoxLayout()
        self.dropdown_box = QComboBox(self)
        self.dropdown_box.addItems(text_selection)  # Add your options here
        self.dropdown_box.setCurrentText("English")
        self.dropdown_box.currentIndexChanged.connect(self.on_user_input_changed)
        self.text_input_layout.addWidget(self.dropdown_box)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter the text you want to be read here...")
        self.text_input.textChanged.connect(self.on_user_input_changed)
        self.text_input_layout.addWidget(self.text_input)
        self.main_layout.insertLayout(0, self.text_input_layout)
        self.text_input.setFocus()
        self.text_input.setText("")

        # Initialize plots
        self.init_plots()

        # Initialize slider and buttons
        self.init_controls()

        # Initialize Timer for TTS Cooldown
        self.tts_timer = QTimer()
        self.tts_timer.setSingleShot(True)
        self.tts_timer.timeout.connect(self.run_tts)
        self.tts_update_required = False

    def clear_all_widgets(self):
        self.spectrogram_view.setParent(None)
        self.pitch_plot.setParent(None)
        self.upper_row.setParent(None)
        self.slider_label.setParent(None)
        self.mod_slider.setParent(None)
        self.slider_value_label.setParent(None)
        self.generate_button.setParent(None)
        self.load_audio_button.setParent(None)
        self.save_audio_button.setParent(None)
        self.play_audio_button.setParent(None)

    def init_plots(self):
        # Spectrogram Plot
        self.spectrogram_view = pg.PlotWidget(background="#f5f5f5")
        self.spectrogram_view.setLabel('left', 'Frequency Buckets', units='')
        self.spectrogram_view.setLabel('bottom', 'Phonemes', units='')
        self.main_layout.addWidget(self.spectrogram_view)

        # Pitch Plot
        self.pitch_plot = pg.PlotWidget(background="#f5f5f5")
        self.pitch_plot.setLabel('left', 'Intonation', units='')
        self.pitch_plot.setLabel('bottom', 'Phonemes', units='')
        self.main_layout.addWidget(self.pitch_plot)

    def load_data(self, durations, pitch, spectrogram):

        durations = remove_indexes(durations, self.word_boundaries)
        pitch = remove_indexes(pitch, self.word_boundaries)

        self.durations = durations
        self.cumulative_durations = np.cumsum(self.durations)
        self.spectrogram = spectrogram

        # Display Spectrogram
        self.spectrogram_view.setLimits(xMin=0, xMax=self.cumulative_durations[-1] + 10, yMin=0, yMax=1000)  # Adjust as per your data
        self.spectrogram_view.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.spectrogram_view.setMouseEnabled(x=False, y=False)  # Disable panning and zooming
        img = pg.ImageItem(self.spectrogram)
        self.spectrogram_view.addItem(img)
        img.setLookupTable(pg.colormap.get('GnBu', source='matplotlib').getLookupTable())
        spectrogram_ticks = self.get_phoneme_ticks(self.cumulative_durations)
        self.spectrogram_view.getAxis('bottom').setTicks([spectrogram_ticks])
        spectrogram_label_color = QColor('#006400')
        self.spectrogram_view.getAxis('bottom').setTextPen(QPen(spectrogram_label_color))
        self.spectrogram_view.getAxis('left').setTextPen(QPen(spectrogram_label_color))

        # Display Pitch
        self.pitch_curve = self.pitch_plot.plot(self.cumulative_durations, self.pitch, pen=pg.mkPen('#B8860B', width=8), name='Pitch')
        self.pitch_plot.setMouseEnabled(x=False, y=False)  # Disable panning and zooming
        pitch_ticks = self.get_phoneme_ticks(self.cumulative_durations)
        self.pitch_plot.getAxis('bottom').setTicks([pitch_ticks])
        pitch_label_color = QColor('#006400')
        self.pitch_plot.getAxis('bottom').setTextPen(QPen(pitch_label_color))
        self.pitch_plot.getAxis('left').setTextPen(QPen(pitch_label_color))

        # Display Durations
        self.duration_lines = []
        for i, cum_dur in enumerate(self.cumulative_durations):
            line = pg.InfiniteLine(pos=cum_dur, angle=90, pen=pg.mkPen('orange', width=4))
            self.spectrogram_view.addItem(line)
            line.setMovable(True)
            # Use lambda with default argument to capture current i
            line.sigPositionChanged.connect(lambda _, idx=i: self.on_duration_changed(idx))
            self.duration_lines.append(line)

        self.enable_interactions()

    def get_phoneme_ticks(self, cumulative_durations):
        """
        Create ticks for phoneme labels centered between durations.
        """
        ticks = []
        previous = 0
        for i, cum_dur in enumerate(cumulative_durations):
            if i == 0:
                center = cum_dur / 2
            else:
                center = (previous + cum_dur) / 2
            ticks.append((center, self.phonemes[i]))
            previous = cum_dur
        return ticks

    def init_controls(self):
        # Main vertical layout for controls
        self.controls_layout = QVBoxLayout()
        self.main_layout.addLayout(self.controls_layout)

        # Upper row layout for slider
        self.upper_row = QHBoxLayout()
        self.controls_layout.addLayout(self.upper_row)

        # Slider Label
        self.slider_label = QLabel("Faster")
        self.upper_row.addWidget(self.slider_label)

        # Slider
        self.mod_slider = QSlider(Qt.Horizontal)
        self.mod_slider.setMinimum(70)
        self.mod_slider.setMaximum(130)
        self.mod_slider.setValue(self.slider_val)
        self.mod_slider.setTickPosition(QSlider.TicksBelow)
        self.mod_slider.setTickInterval(10)
        self.mod_slider.valueChanged.connect(self.on_slider_changed)
        self.upper_row.addWidget(self.mod_slider)

        # Slider Value Display
        self.slider_value_label = QLabel("Slower")
        self.upper_row.addWidget(self.slider_value_label)

        # Lower row layout for buttons
        self.lower_row = QHBoxLayout()
        self.controls_layout.addLayout(self.lower_row)

        self.generate_button = QPushButton("Generate new Prosody")
        self.generate_button.clicked.connect(self.generate_new_prosody)
        self.lower_row.addWidget(self.generate_button)

        self.load_audio_button = QPushButton("Load Example of Voice to Mimic")
        self.load_audio_button.clicked.connect(self.load_audio_file)
        self.lower_row.addWidget(self.load_audio_button)

        self.save_audio_button = QPushButton("Save Audio File")
        self.save_audio_button.clicked.connect(self.save_audio_file)
        self.lower_row.addWidget(self.save_audio_button)

        self.play_audio_button = QPushButton("Play Audio")
        self.play_audio_button.clicked.connect(self.play_audio)
        self.lower_row.addWidget(self.play_audio_button)

    def enable_interactions(self):
        x_pitch = self.cumulative_durations.copy()
        y_pitch = self.pitch.copy()
        self.pitch_scatter = DraggableScatter(x_pitch,
                                              y_pitch,
                                              pen=pg.mkPen(None),
                                              brush=pg.mkBrush(218, 165, 32, 250),  # Pastel accent color
                                              size=25, )
        self.pitch_scatter.pointMoved.connect(self.on_pitch_point_moved)
        self.pitch_plot.addItem(self.pitch_scatter)

    def on_duration_changed(self, idx):
        """
        Moving a duration line adjusts the position of that line and all subsequent lines.
        Ensures that durations do not become negative.
        """
        min_duration = self.min_duration

        # Get new position of the moved line
        new_pos = self.duration_lines[idx].value()

        # Calculate the minimum allowed position
        if idx == 0:
            min_allowed = min_duration
        else:
            min_allowed = self.duration_lines[idx - 1].value() + min_duration

        # Clamp new_pos
        if new_pos < min_allowed:
            new_pos = min_allowed

        # If the new_pos was clamped, update the line's position without emitting signal again
        if new_pos != self.duration_lines[idx].value():
            self.duration_lines[idx].blockSignals(True)
            self.duration_lines[idx].setValue(new_pos)
            self.duration_lines[idx].blockSignals(False)

        # Calculate the delta change
        delta = new_pos - self.cumulative_durations[idx]

        # Update current and subsequent cumulative durations
        for i in range(idx, len(self.cumulative_durations)):
            self.cumulative_durations[i] += delta
            self.duration_lines[i].blockSignals(True)
            self.duration_lines[i].setValue(self.cumulative_durations[i])
            self.duration_lines[i].blockSignals(False)

        # Update durations based on cumulative durations
        self.durations = np.diff(np.insert(self.cumulative_durations, 0, 0)).tolist()

        # print(f"Updated Durations: {self.durations}")

        # Update pitch curve
        self.pitch_curve.setData(self.cumulative_durations, self.pitch)

        # Update pitch scatter points
        self.pitch_scatter.setData(x=self.cumulative_durations, y=self.pitch)
        self.pitch_scatter.x = self.cumulative_durations

        # Update phoneme ticks
        spectrogram_ticks = self.get_phoneme_ticks(self.cumulative_durations)
        self.spectrogram_view.getAxis('bottom').setTicks([spectrogram_ticks])
        self.pitch_plot.getAxis('bottom').setTicks([spectrogram_ticks])

        # Update spectrogram's X-axis limits
        self.spectrogram_view.setLimits(xMin=0, xMax=self.cumulative_durations[-1] + 10)  # Added buffer

        # Mark that an update is required
        self.mark_tts_update()

    def on_pitch_point_moved(self, index, new_y):
        # Update the pitch array with the new y-value
        self.pitch[index] = new_y
        # print(f"Pitch point {index} moved to {new_y:.2f} Hz")
        # Update the pitch curve line
        self.pitch_curve.setData(self.cumulative_durations, self.pitch)
        # Update the scatter points' y-values (x remains the same)
        self.pitch_scatter.y[index] = new_y
        self.pitch_scatter.setData(x=self.cumulative_durations, y=self.pitch)
        # Mark that an update is required
        self.mark_tts_update()

    def on_user_input_changed(self, text):
        """
        Handle changes in the text input field.
        """
        # print(f"User input changed: {text}")
        # Mark that an update is required
        self.mark_tts_update()

    def on_slider_changed(self, value):
        # Update the slider label
        # self.slider_value_label.setText(f"Durations at {value}%")
        self.slider_val = value
        # print(f"Slider changed to {scaling_factor * 100}% speed")
        # Mark that an update is required
        self.mark_tts_update()

    def generate_new_prosody(self):
        """
        Generate new prosody.
        """
        wave, mel, durations, pitch = self.tts_backend(text=self.text_input.text(),
                                                       view=False,
                                                       duration_scaling_factor=1.0,
                                                       pitch_variance_scale=1.0,
                                                       energy_variance_scale=1.0,
                                                       pause_duration_scaling_factor=1.0,
                                                       durations=None,
                                                       pitch=None,
                                                       energy=None,
                                                       input_is_phones=False,
                                                       return_plot_as_filepath=False,
                                                       loudness_in_db=-29.0,
                                                       prosody_creativity=0.8,
                                                       return_everything=True)
        # reset and clear everything
        self.slider_val = 100
        self.prev_slider_val_for_denorm = self.slider_val
        self.durations_are_scaled = False
        self.clear_all_widgets()
        self.init_plots()
        self.init_controls()

        self.load_data(durations=durations.cpu().numpy(), pitch=pitch.cpu().numpy(), spectrogram=mel.cpu().transpose(0, 1).numpy())

        self.update_result_audio(wave)
        self.cumulative_durations = np.cumsum(self.durations)

        # Update scatter points
        self.pitch_scatter.setData(x=self.cumulative_durations, y=self.pitch)

        # Update curves
        self.pitch_curve.setData(self.cumulative_durations, self.pitch)

        # Update duration lines positions
        for i, line in enumerate(self.duration_lines):
            line.blockSignals(True)
            line.setValue(self.cumulative_durations[i])
            line.blockSignals(False)

        # Update phoneme ticks
        spectrogram_ticks = self.get_phoneme_ticks(self.cumulative_durations)
        self.spectrogram_view.getAxis('bottom').setTicks([spectrogram_ticks])
        self.pitch_plot.getAxis('bottom').setTicks([spectrogram_ticks])

        # print("Generated new random prosody.")

    def load_audio_file(self):
        """
        Open a file dialog to load an audio file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_filter = "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Example of Voice to Mimic", "", file_filter, options=options)
        if file_path:
            self.audio_file_path = file_path
            # print(f"Loaded audio file: {self.audio_file_path}")
            # Here, you can add code to process the loaded audio if needed
            self.mark_tts_update()

    def save_audio_file(self):
        """
        Open a file dialog to save the resulting audio NumPy array.
        """
        if self.result_audio is None:
            QMessageBox.warning(self, "Save Error", "No resulting audio to save.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_filter = "WAV Files (*.wav);;All Files (*)"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", file_filter, options=options)
        if save_path:
            try:
                sample_rate = 24000

                # Normalize the audio if it's not in the correct range
                if self.result_audio.dtype != np.int16:
                    audio_normalized = np.int16(self.result_audio / np.max(np.abs(self.result_audio)) * 32767)
                else:
                    audio_normalized = self.result_audio

                # Save using scipy.io.wavfile
                scipy.io.wavfile.write(save_path, sample_rate, audio_normalized)
                # print(f"Audio saved successfully at: {save_path}")
                QMessageBox.information(self, "Save Successful", f"Audio saved successfully at:\n{save_path}")
            except Exception as e:
                print(f"Error saving audio: {e}")
                QMessageBox.critical(self, "Save Error", f"Failed to save audio:\n{e}")

    def play_audio(self):
        # print("playing current audio...")
        sounddevice.play(self.result_audio, samplerate=24000)

    def update_result_audio(self, audio_array):
        """
        Update the resulting audio array.
        This method should be called with your TTS model's output.
        """
        self.result_audio = audio_array
        # print("Resulting audio updated.")

    def mark_tts_update(self):
        """
        Marks that a TTS update is required and starts/resets the timer.
        """
        self.tts_update_required = True
        self.tts_timer.start(600)  # 600 milliseconds

    def run_tts(self):
        """
        Dummy method to simulate running the TTS model.
        This should be replaced with actual TTS integration.
        """
        text = self.text_input.text()
        while self.tts_update_required:
            self.tts_update_required = False
            if text.strip() == "":
                return

            # print(f"Running TTS with text: {text}")

            # reset and clear everything
            self.clear_all_widgets()
            self.init_plots()
            self.init_controls()

            if self.audio_file_path is not None:
                self.tts_backend.set_utterance_embedding(self.audio_file_path)

            self.tts_backend.set_language(self.name_to_iso[self.dropdown_box.currentText()])

            phonemes = self.tts_backend.text2phone.get_phone_string(text=text)
            self.phonemes = phonemes.replace(" ", "")

            forced_durations = None if self.durations is None or len(self.durations) != len(self.phonemes) else insert_zeros_at_indexes(self.durations, self.word_boundaries)
            if forced_durations is not None and self.durations_are_scaled:
                forced_durations = torch.LongTensor([forced_duration / (self.prev_slider_val_for_denorm / 100) for forced_duration in forced_durations]).unsqueeze(0)  # revert scaling
            elif forced_durations is not None:
                forced_durations = torch.LongTensor(forced_durations).unsqueeze(0)
            forced_pitch = None if self.pitch is None or len(self.pitch) != len(self.phonemes) else torch.tensor(insert_zeros_at_indexes(self.pitch, self.word_boundaries)).unsqueeze(0)

            wave, mel, durations, pitch = self.tts_backend(text,
                                                           view=False,
                                                           duration_scaling_factor=self.slider_val / 100,
                                                           pitch_variance_scale=1.0,
                                                           energy_variance_scale=1.0,
                                                           pause_duration_scaling_factor=1.0,
                                                           durations=forced_durations,
                                                           pitch=forced_pitch,
                                                           energy=None,
                                                           input_is_phones=False,
                                                           return_plot_as_filepath=False,
                                                           loudness_in_db=-29.0,
                                                           prosody_creativity=0.1,
                                                           return_everything=True)

            self.word_boundaries = find_zero_indexes(durations)
            self.prev_slider_val_for_denorm = self.slider_val
            if self.slider_val != 100:
                self.durations_are_scaled = True

            self.load_data(durations=durations.cpu().numpy(), pitch=pitch.cpu().numpy(), spectrogram=mel.cpu().transpose(0, 1).numpy())

            self.update_result_audio(wave)
            # print("TTS run completed and plots/audio updated.")


def main():
    app = QApplication(sys.argv)
    stylesheet = """
    QMainWindow {
        background-color: #f5f5f5;
        color: #333333;
        font-family: system-ui;
    }

    QWidget {
        background-color: #f5f5f5;
        color: #333333;
        font-size: 14px;
    }

    QPushButton {
        background-color: #808000;  
        border: 1px solid #ffffff;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 10px;
    }

    QPushButton:hover {
        background-color: #228B22;
    }

    QPushButton:pressed {
        background-color: #006400;  
    }

    QSlider::groove:horizontal {
        border: 1px solid #bbb;
        background: #d3d3d3;
        height: 8px;
        border-radius: 4px;
    }

    QSlider::handle:horizontal {
        background: #D2691E;  
        border: 1px solid #D2691E;
        width: 26px;
        margin: -5px 0;
        border-radius: 9px;
    }

    QLabel {
        color: #006400;
    }

    QLineEdit {
        background-color: #EEE8AA;
        border: 10px solid #DAA520;
        padding: 12px;
        border-radius: 20px;
    }

    QLineEdit:focus {
        background-color: #EEE8AA;
        border: 10px solid #DAA520;
        padding: 12px;
        border-radius: 20px;    
    }
    """
    app.setStyleSheet(stylesheet)

    interface = TTSInterface(ToucanTTSInterface(device="cuda" if torch.cuda.is_available() else "cpu"))
    interface.show()
    sys.exit(app.exec_())


def find_zero_indexes(numbers):
    zero_indexes = [index for index, value in enumerate(numbers) if value == 0]
    return zero_indexes


def remove_indexes(data_list, indexes_to_remove):
    result = [value for i, value in enumerate(data_list) if i not in indexes_to_remove]
    return result


def insert_zeros_at_indexes(data_list, indexes_to_add_zeros):
    if len(indexes_to_add_zeros) == 0:
        return data_list
    result = data_list[:]
    for index in sorted(indexes_to_add_zeros):
        result.insert(index, 0)
    return result


if __name__ == "__main__":
    main()
