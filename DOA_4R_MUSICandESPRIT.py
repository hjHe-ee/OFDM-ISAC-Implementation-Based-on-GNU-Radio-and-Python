#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt, QtCore
import sip
import sys
import signal
import numpy as np
import threading
import traceback

from gnuradio import gr
from gnuradio import qtgui
from gnuradio import blocks


class SafeApplication(Qt.QApplication):
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            return False


def ula_steering_vec(n_rx: int, theta_rad: float) -> np.ndarray:
    m = np.arange(n_rx, dtype=np.float64)
    return np.exp(-1j * np.pi * m * np.sin(theta_rad)).astype(np.complex64)


def ula_steering_mat(n_rx: int, theta_rad_grid: np.ndarray) -> np.ndarray:
    m = np.arange(n_rx, dtype=np.float64)[:, None]
    s = np.sin(theta_rad_grid.astype(np.float64))[None, :]
    A = np.exp(-1j * np.pi * m * s)
    return A.astype(np.complex64)


def music_1src_spectrum(R: np.ndarray, A_scan: np.ndarray) -> tuple[np.ndarray, int]:
    R = (R + R.conj().T) * 0.5
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    En = V[:, 1:]
    proj = En.conj().T @ A_scan
    den = np.sum(np.abs(proj) ** 2, axis=0).astype(np.float64) + 1e-12
    P = 1.0 / den
    k_peak = int(np.argmax(P))
    return P, k_peak


def esprit_1src_angle_deg(R: np.ndarray) -> float:
    R = (R + R.conj().T) * 0.5
    w, V = np.linalg.eigh(R)
    u = V[:, np.argsort(w)[::-1][0]].astype(np.complex64)

    u1 = u[:-1].reshape(-1, 1)
    u2 = u[1:].reshape(-1, 1)

    den = (u1.conj().T @ u1)[0, 0]
    if np.abs(den) < 1e-12:
        return 0.0
    phi = (u1.conj().T @ u2)[0, 0] / den

    phase = float(np.angle(phi))
    s = -phase / np.pi
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.rad2deg(np.arcsin(s)))


class DoAManager:
    def __init__(self, n_rx=4, n_scan=181, seed=2026):
        self.lock = threading.RLock()
        self.n_rx = int(n_rx)
        self.n_scan = int(n_scan)

        self.ang_grid_deg = np.linspace(-90.0, 90.0, self.n_scan, dtype=np.float64)
        self.ang_grid_rad = np.deg2rad(self.ang_grid_deg)
        self.A_scan = ula_steering_mat(self.n_rx, self.ang_grid_rad)

        self.rng = np.random.default_rng(seed)

        self.target_aoa_deg = 20.0
        self.snr_db = 20.0

        self.epoch = 0
        self.last_music_deg = float("nan")
        self.last_esprit_deg = float("nan")

    def set_target_aoa_deg(self, v: float):
        with self.lock:
            v = float(v)
            if abs(v - self.target_aoa_deg) > 1e-9:
                self.target_aoa_deg = v
                self.epoch += 1

    def set_snr_db(self, v: float):
        with self.lock:
            v = float(v)
            if abs(v - self.snr_db) > 1e-9:
                self.snr_db = v
                self.epoch += 1

    def set_last_estimates(self, music_deg: float, esprit_deg: float):
        with self.lock:
            self.last_music_deg = float(music_deg)
            self.last_esprit_deg = float(esprit_deg)

    def get_status(self):
        with self.lock:
            return (float(self.target_aoa_deg), float(self.snr_db),
                    float(self.last_music_deg), float(self.last_esprit_deg))

    def get_epoch_and_params(self):
        with self.lock:
            return int(self.epoch), float(self.target_aoa_deg), float(self.snr_db)

    def gen_array_snapshots(self, L: int, aoa_deg: float, snr_db: float) -> np.ndarray:
        a = ula_steering_vec(self.n_rx, np.deg2rad(float(aoa_deg)))

        s = (self.rng.normal(0, 1/np.sqrt(2), size=L) +
             1j * self.rng.normal(0, 1/np.sqrt(2), size=L)).astype(np.complex64)

        X = (a[:, None] * s[None, :]).astype(np.complex64)

        sig_pow = float(np.mean(np.abs(X) ** 2) + 1e-12)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_pow = sig_pow / snr_lin

        n = (self.rng.normal(0, np.sqrt(noise_pow/2), size=X.shape) +
             1j * self.rng.normal(0, np.sqrt(noise_pow/2), size=X.shape)).astype(np.complex64)

        return (X + n).astype(np.complex64)

    def gen_rx0_stream(self, n: int) -> np.ndarray:
        _, _, snr_db = self.get_epoch_and_params()

        s = (self.rng.normal(0, 1/np.sqrt(2), size=n) +
             1j * self.rng.normal(0, 1/np.sqrt(2), size=n)).astype(np.complex64)
        x = s.astype(np.complex64)

        sig_pow = float(np.mean(np.abs(x) ** 2) + 1e-12)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_pow = sig_pow / snr_lin

        n0 = (self.rng.normal(0, np.sqrt(noise_pow/2), size=n) +
              1j * self.rng.normal(0, np.sqrt(noise_pow/2), size=n)).astype(np.complex64)

        return (x + n0).astype(np.complex64)


class rx0_stream_source(gr.sync_block):
    def __init__(self, mgr: DoAManager):
        self.mgr = mgr
        gr.sync_block.__init__(self, name="rx0_stream_source", in_sig=None, out_sig=[np.complex64])

    def work(self, input_items, output_items):
        out = output_items[0]
        out[:] = self.mgr.gen_rx0_stream(len(out))
        return len(out)


class doa_spectrum_source(gr.sync_block):
    def __init__(self, mgr: DoAManager, n_snap=128, batch_L=64, floor_db=-40.0):
        self.mgr = mgr
        self.n_scan = mgr.n_scan
        self.n_rx = mgr.n_rx

        self.n_snap = int(n_snap)
        self.batch_L = int(batch_L)

        self.floor_db = float(floor_db)   # 关键：把谱底抬到 -40dB，曲线才看得见
        self.Xbuf = np.zeros((self.n_rx, 0), dtype=np.complex64)
        self.last_epoch = -1

        gr.sync_block.__init__(
            self,
            name="doa_spectrum_source",
            in_sig=None,
            out_sig=[
                (np.float32, self.n_scan),
                (np.float32, self.n_scan),
            ],
        )

    def _reset(self):
        self.Xbuf = np.zeros((self.n_rx, 0), dtype=np.complex64)

    def set_n_snap(self, v):
        self.n_snap = int(max(16, min(512, int(v))))
        self._reset()

    def work(self, input_items, output_items):
        spec_db = output_items[0]
        mark_db = output_items[1]

        epoch, aoa_deg, snr_db = self.mgr.get_epoch_and_params()
        if epoch != self.last_epoch:
            self._reset()
            self.last_epoch = epoch

        Xb = self.mgr.gen_array_snapshots(self.batch_L, aoa_deg, snr_db)
        self.Xbuf = np.concatenate([self.Xbuf, Xb], axis=1)
        if self.Xbuf.shape[1] > self.n_snap:
            self.Xbuf = self.Xbuf[:, -self.n_snap:]

        if self.Xbuf.shape[1] < max(16, self.n_snap // 4):
            spec_db[0, :] = np.float32(self.floor_db)
            mark_db[0, :] = np.float32(self.floor_db)
            return 1

        X = self.Xbuf
        L = X.shape[1]
        R = (X @ X.conj().T) / float(L)

        tr = float(np.trace(R).real) + 1e-12
        R = R + (1e-3 * (tr / self.n_rx)) * np.eye(self.n_rx, dtype=np.complex64)

        P_lin, k_peak = music_1src_spectrum(R, self.mgr.A_scan)
        peak_ang = float(self.mgr.ang_grid_deg[k_peak])

        # 归一化 -> dB，并把 floor 抬到 -40dB（曲线清晰可见）
        P_db = 10.0 * np.log10(P_lin / (np.max(P_lin) + 1e-12) + 1e-12)
        P_db = np.clip(P_db, self.floor_db, 0.0).astype(np.float32)
        spec_db[0, :] = P_db

        est_ang = float(esprit_1src_angle_deg(R))
        mk = (self.floor_db * np.ones_like(P_db)).astype(np.float32)
        idx = int(np.argmin(np.abs(self.mgr.ang_grid_deg - est_ang)))
        mk[idx] = 0.0
        mark_db[0, :] = mk

        self.mgr.set_last_estimates(peak_ang, est_ang)
        return 1


def make_vector_sink_f(vlen, x_start, x_step, title, nconn=1):
    return qtgui.vector_sink_f(
        int(vlen),
        float(x_start),
        float(x_step),
        "Angle (deg)",
        "Spectrum (dB)",
        str(title),
        int(nconn),
        None
    )


def _try_disable_hold_and_set_axes(vsink, y_min=-40.0, y_max=2.0):
    # 1) 固定 y 轴范围，避免默认 -10..10 导致看不到 -80 的曲线
    if hasattr(vsink, "set_y_axis"):
        try:
            vsink.set_y_axis(float(y_min), float(y_max))
        except Exception:
            pass

    if hasattr(vsink, "enable_autoscale"):
        try:
            vsink.enable_autoscale(False)
        except Exception:
            pass

    # 2) 尝试关闭 max/min hold（不同版本方法名可能不同，逐个 try）
    for fn in ("enable_max_hold", "enable_min_hold"):
        if hasattr(vsink, fn):
            try:
                getattr(vsink, fn)(False)
            except Exception:
                pass


def _try_clear_sink(vsink):
    # 清历史/hold（不同版本可能是 clear_data/reset）
    for fn in ("clear_data", "reset"):
        if hasattr(vsink, fn):
            try:
                getattr(vsink, fn)()
                return
            except Exception:
                pass


class ula4_doa_gui(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "ULA4 DOA (MUSIC+ESPRIT)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("4Rx ULA DOA - Rx time + MUSIC Spectrum + ESPRIT Marker (Windows stable)")
        qtgui.util.check_set_qss()

        self.n_scan = 181
        self.mgr = DoAManager(n_rx=4, n_scan=self.n_scan, seed=2026)

        self.top_scroll_layout = Qt.QVBoxLayout(self)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)

        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)

        # ---- controls
        ctrl_box = Qt.QGroupBox("Controls")
        ctrl_layout = Qt.QGridLayout(ctrl_box)

        self.aoa_slider = Qt.QSlider(QtCore.Qt.Horizontal)
        self.aoa_slider.setMinimum(-160)
        self.aoa_slider.setMaximum(160)
        self.aoa_slider.setValue(int(round(20.0 / 0.5)))

        self.aoa_spin = Qt.QDoubleSpinBox()
        self.aoa_spin.setRange(-80.0, 80.0)
        self.aoa_spin.setSingleStep(0.5)
        self.aoa_spin.setDecimals(1)
        self.aoa_spin.setValue(20.0)

        ctrl_layout.addWidget(Qt.QLabel("Target AOA (deg)"), 0, 0)
        ctrl_layout.addWidget(self.aoa_slider, 0, 1)
        ctrl_layout.addWidget(self.aoa_spin, 0, 2)

        self.snr_slider = Qt.QSlider(QtCore.Qt.Horizontal)
        self.snr_slider.setMinimum(int(round(-5.0 / 0.5)))
        self.snr_slider.setMaximum(int(round(40.0 / 0.5)))
        self.snr_slider.setValue(int(round(20.0 / 0.5)))

        self.snr_spin = Qt.QDoubleSpinBox()
        self.snr_spin.setRange(-5.0, 40.0)
        self.snr_spin.setSingleStep(0.5)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setValue(20.0)

        ctrl_layout.addWidget(Qt.QLabel("SNR (dB)"), 1, 0)
        ctrl_layout.addWidget(self.snr_slider, 1, 1)
        ctrl_layout.addWidget(self.snr_spin, 1, 2)

        self.snap_slider = Qt.QSlider(QtCore.Qt.Horizontal)
        self.snap_slider.setMinimum(16)
        self.snap_slider.setMaximum(512)
        self.snap_slider.setValue(128)

        self.snap_spin = Qt.QSpinBox()
        self.snap_spin.setRange(16, 512)
        self.snap_spin.setSingleStep(1)
        self.snap_spin.setValue(128)

        ctrl_layout.addWidget(Qt.QLabel("Snapshots (SCM length)"), 2, 0)
        ctrl_layout.addWidget(self.snap_slider, 2, 1)
        ctrl_layout.addWidget(self.snap_spin, 2, 2)

        self.top_layout.addWidget(ctrl_box)

        self.lbl_status = Qt.QLabel("Target=20.0°, SNR=20.0 dB | MUSIC=nan°, ESPRIT=nan°")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        self.top_layout.addWidget(self.lbl_status)

        self.status_timer = Qt.QTimer()
        self.status_timer.timeout.connect(self._refresh_status)
        self.status_timer.start(200)

        # ---- Rx0 time plot
        self.rx_rate = 20000.0
        self.rx_src = rx0_stream_source(self.mgr)
        self.rx_th = blocks.throttle(gr.sizeof_gr_complex, self.rx_rate, True)

        self.rx_time = qtgui.time_sink_c(1024, self.rx_rate, "Rx0 Time-domain (complex)", 1, None)
        self.rx_time.set_update_time(0.10)
        self.rx_time.enable_autoscale(True)
        self._rx_time_win = sip.wrapinstance(self.rx_time.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._rx_time_win)

        # ---- Spectrum source + plot
        self.est_rate = 10000.0
        self.floor_db = -30.0

        self.spec_src = doa_spectrum_source(self.mgr, n_snap=128, batch_L=64, floor_db=self.floor_db)

        self.th_spec = blocks.throttle(gr.sizeof_float * self.n_scan, self.est_rate, True)
        self.th_mark = blocks.throttle(gr.sizeof_float * self.n_scan, self.est_rate, True)

        self.spec_sink = make_vector_sink_f(
            self.n_scan, -90.0, 1.0, "Spatial Spectrum: MUSIC (curve) + ESPRIT (marker)", nconn=2
        )
        self.spec_sink.set_update_time(0.05)
        _try_disable_hold_and_set_axes(self.spec_sink, y_min=self.floor_db, y_max=2.0)

        self._spec_sink_win = sip.wrapinstance(self.spec_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._spec_sink_win)

        # ---- connections
        self.connect((self.rx_src, 0), (self.rx_th, 0))
        self.connect((self.rx_th, 0), (self.rx_time, 0))

        self.connect((self.spec_src, 0), (self.th_spec, 0))
        self.connect((self.spec_src, 1), (self.th_mark, 0))
        self.connect((self.th_spec, 0), (self.spec_sink, 0))
        self.connect((self.th_mark, 0), (self.spec_sink, 1))

        # ---- wire up controls
        self.aoa_slider.valueChanged.connect(self._on_aoa_slider)
        self.aoa_spin.valueChanged.connect(self._on_aoa_spin)

        self.snr_slider.valueChanged.connect(self._on_snr_slider)
        self.snr_spin.valueChanged.connect(self._on_snr_spin)

        self.snap_slider.valueChanged.connect(self._on_snap_slider)
        self.snap_spin.valueChanged.connect(self._on_snap_spin)

        self.mgr.set_target_aoa_deg(20.0)
        self.mgr.set_snr_db(20.0)

    def _refresh_status(self):
        tgt, snr, m, e = self.mgr.get_status()
        self.lbl_status.setText(
            f"Target={tgt:.1f}°, SNR={snr:.1f} dB | MUSIC={m:.2f}°, ESPRIT={e:.2f}°"
        )

    def _clear_plot(self):
        _try_clear_sink(self.spec_sink)

    def _on_aoa_slider(self, iv):
        v = float(iv) * 0.5
        self.aoa_spin.blockSignals(True)
        self.aoa_spin.setValue(v)
        self.aoa_spin.blockSignals(False)
        self.mgr.set_target_aoa_deg(v)
        self._clear_plot()   # 清掉 hold/历史，视觉上立刻更新

    def _on_aoa_spin(self, v):
        iv = int(round(float(v) / 0.5))
        self.aoa_slider.blockSignals(True)
        self.aoa_slider.setValue(iv)
        self.aoa_slider.blockSignals(False)
        self.mgr.set_target_aoa_deg(float(v))
        self._clear_plot()

    def _on_snr_slider(self, iv):
        v = float(iv) * 0.5
        self.snr_spin.blockSignals(True)
        self.snr_spin.setValue(v)
        self.snr_spin.blockSignals(False)
        self.mgr.set_snr_db(v)
        self._clear_plot()

    def _on_snr_spin(self, v):
        iv = int(round(float(v) / 0.5))
        self.snr_slider.blockSignals(True)
        self.snr_slider.setValue(iv)
        self.snr_slider.blockSignals(False)
        self.mgr.set_snr_db(float(v))
        self._clear_plot()

    def _on_snap_slider(self, iv):
        self.snap_spin.blockSignals(True)
        self.snap_spin.setValue(int(iv))
        self.snap_spin.blockSignals(False)
        self.spec_src.set_n_snap(int(iv))
        self._clear_plot()

    def _on_snap_spin(self, v):
        self.snap_slider.blockSignals(True)
        self.snap_slider.setValue(int(v))
        self.snap_slider.blockSignals(False)
        self.spec_src.set_n_snap(int(v))
        self._clear_plot()

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


def main():
    qapp = SafeApplication(sys.argv)
    tb = ula4_doa_gui()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()
