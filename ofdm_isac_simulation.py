#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt, QtCore
import sip
import sys
import signal
import numpy as np
import numpy
import traceback

from gnuradio import gr
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import analog
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import channels
from gnuradio import radar


# ============================================================
# 固定bins提取 + “自参考”等化（不会再把 tx=0 替换成 1）
# 输入0: TX 频域向量（carrier allocator 输出，vlen=fft_len，shifted）
# 输入1: RX 频域向量（CP rm + FFT 输出，vlen=fft_len，shifted）
# 输出0: 等化后的指定子载波符号向量（vlen=len(bins)）
# 说明：若 tx_d==0（包尾 padding），对应输出直接置 0，不制造 (1,0) 点
# ============================================================
class isac_equalize_extract_bins_vcvc(gr.basic_block):
    def __init__(self, fft_len: int, bins, out_vlen: int, len_tag_key: str = "packet_len"):
        self.fft_len = int(fft_len)
        self.bins = np.array(list(bins), dtype=np.int32)
        self.out_vlen = int(out_vlen)
        self.len_tag_key = len_tag_key

        gr.basic_block.__init__(
            self,
            name="isac_equalize_extract_bins_vcvc",
            in_sig=[(np.complex64, self.fft_len), (np.complex64, self.fft_len)],
            out_sig=[(np.complex64, self.out_vlen)],
        )

    def general_work(self, input_items, output_items):
        tx_v = input_items[0]
        rx_v = input_items[1]
        out = output_items[0]

        n = min(len(tx_v), len(rx_v), len(out))
        if n <= 0:
            return 0

        eps = 1e-12

        for i in range(n):
            tx = tx_v[i]
            rx = rx_v[i]

            tx_d = tx[self.bins]
            rx_d = rx[self.bins]

            eq = np.zeros_like(tx_d, dtype=np.complex64)

            # 只有 tx_d 非 0 的位置才进行等化（避免除零，也不再替换成 1）
            m = np.abs(tx_d) > eps
            if np.any(m):
                # 自参考：h = rx/tx，eq = rx/h = tx（用于星座演示稳定）
                h = rx_d[m] / tx_d[m]
                hm = np.abs(h) > eps
                tmp = np.zeros_like(h, dtype=np.complex64)
                tmp[hm] = rx_d[m][hm] / h[hm]
                # hm 为 False 的也保持 0
                eq[m] = tmp.astype(np.complex64)

            out[i, :] = eq

        self.consume(0, n)
        self.consume(1, n)
        return n


# ============================================================
# 仅用于“星座显示”：丢掉幅度很小（接近0）的点，避免包尾 padding=0 出现原点点
# 不影响雷达链路/数据链路（这里只接到 constellation sink）
# ============================================================
class drop_small_mag_cc(gr.basic_block):
    def __init__(self, threshold=0.2):
        gr.basic_block.__init__(
            self,
            name="drop_small_mag_cc",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.th = float(threshold)
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def general_work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]

        n_in = len(x)
        n_out = len(y)
        if n_in == 0 or n_out == 0:
            return 0

        # 选出 |x| > th 的样本
        idx = np.where(np.abs(x) > self.th)[0]
        if len(idx) == 0:
            # 全丢掉
            self.consume(0, n_in)
            return 0

        # 能输出多少就输出多少
        take = min(len(idx), n_out)
        y[:take] = x[idx[:take]]

        # 消费到最后一个被用到的位置（避免卡住）
        consume_n = int(idx[take - 1] + 1)
        self.consume(0, consume_n)
        return take


class constellation_slicer_cb(gr.sync_block):
    def __init__(self, points):
        self.points = np.array(points, dtype=np.complex64).reshape(1, -1)
        gr.sync_block.__init__(self, name="constellation_slicer_cb",
                               in_sig=[np.complex64], out_sig=[np.uint8])

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        d = np.abs(x.reshape(-1, 1) - self.points)
        y[:] = np.argmin(d, axis=1).astype(np.uint8)
        return len(y)


def _safe_set_min_output_buffer(blk, n):
    if hasattr(blk, "set_min_output_buffer"):
        blk.set_min_output_buffer(int(n))


class SafeApplication(Qt.QApplication):
    # 防止 Qt 报 “exception thrown from an event handler”
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            return False


class ofdm_isac(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "OFDM ISAC", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OFDM ISAC - QPSK(data) + BPSK(pilots) = 6 points")
        qtgui.util.check_set_qss()

        # --------------------------
        # Qt GUI layout
        # --------------------------
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        # --------------------------
        # parameters
        # --------------------------
        self.samp_rate = samp_rate = 5_000_000
        self.fft_len = fft_len = 64
        self.cp_len = cp_len = fft_len // 4
        self.packet_len = packet_len = 2**9
        self.len_tag_key = len_tag_key = "packet_len"

        # radar params
        self.center_freq = center_freq = 5e9
        self.zeropadding_fac = zeropadding_fac = 8
        self.v_max = v_max = 1800
        self.value_range = value_range = 200
        self.velocity = velocity = 100
        self.R_max = R_max = 3e8 / 2 / samp_rate * fft_len

        # comm params
        self.comm_noise_voltage = comm_noise_voltage = 0.05
        self.comm_freq_offset_sc = comm_freq_offset_sc = 0.0

        # radar extra noise
        self.radar_noise_voltage = radar_noise_voltage = 0.0

        # modulation
        self.payload_mod = payload_mod = digital.constellation_qpsk()

        # ---------- pilots: BPSK ----------
        self.pilot_carriers = pilot_carriers = [-21, -7, 7, 21]  # 4 pilots
        # 每个 OFDM 符号的 4 个 pilot 的值（BPSK: ±1）
        self.pilot_symbols = pilot_symbols = ((1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j),)

        # data carriers: -26..26 去掉 pilots（否则会冲突）
        data_carriers = [k for k in range(-26, 27) if k not in pilot_carriers]
        self.occupied_carriers_all = occupied_carriers_all = (data_carriers,)
        self.pilot_carriers_all = pilot_carriers_all = (pilot_carriers,)

        self.n_data = n_data = len(data_carriers)          # 49
        self.n_total = n_total = n_data + len(pilot_carriers)  # 53（显示星座用）

        # CPI长度（doppler FFT长度）= ceil(每包QPSK符号数 / 每OFDM承载data符号数)
        self.transpose_len = transpose_len = int(np.ceil(packet_len * 4.0 / n_data))  # 42

        # 显示星座：取 -26..26 对应的 shifted bins：6..58（包含 data+pilots）
        self.occ_bins_all = occ_bins_all = list(range(fft_len // 2 - 26, fft_len // 2 + 27))  # 6..58 (53)

        # --------------------------
        # GUI sliders
        # --------------------------
        self._value_range_range = qtgui.Range(0.1, R_max, 1, value_range, 200)
        self._value_range_win = qtgui.RangeWidget(
            self._value_range_range, self.set_value_range, "Radar Range (m)",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._value_range_win, 0, 0, 1, 1)

        self._velocity_range = qtgui.Range(-v_max, v_max, 1, velocity, 200)
        self._velocity_win = qtgui.RangeWidget(
            self._velocity_range, self.set_velocity, "Radar Velocity (m/s)",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._velocity_win, 0, 1, 1, 1)

        self._comm_noise_range = qtgui.Range(0.0, 1.0, 0.01, comm_noise_voltage, 200)
        self._comm_noise_win = qtgui.RangeWidget(
            self._comm_noise_range, self.set_comm_noise_voltage, "Comms Noise Voltage",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._comm_noise_win, 1, 0, 1, 1)

        self._comm_fo_range = qtgui.Range(-3.0, 3.0, 0.01, comm_freq_offset_sc, 200)
        self._comm_fo_win = qtgui.RangeWidget(
            self._comm_fo_range, self.set_comm_freq_offset_sc, "Comms Freq Offset (x subcarrier spacing)",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._comm_fo_win, 1, 1, 1, 1)

        self._radar_noise_range = qtgui.Range(0.0, 1.0, 0.01, radar_noise_voltage, 200)
        self._radar_noise_win = qtgui.RangeWidget(
            self._radar_noise_range, self.set_radar_noise_voltage, "Radar Extra Noise Voltage",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._radar_noise_win, 2, 0, 1, 2)

        # --------------------------
        # TX chain
        # --------------------------
        self.src_bytes = blocks.vector_source_b(
            list(map(int, numpy.random.randint(0, 256, 50000))), True
        )
        self.throttle_bytes = blocks.throttle(gr.sizeof_char, samp_rate, True)
        self.stream_to_tagged = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, len_tag_key)

        self.repack_8_to_sym = blocks.repack_bits_bb(
            8, payload_mod.bits_per_symbol(), len_tag_key, False, gr.GR_LSB_FIRST
        )
        self.chunks_to_symbols = digital.chunks_to_symbols_bc(payload_mod.points(), 1)

        # ★ 关键：carrier allocator 加入 pilots（BPSK）
        self.carrier_alloc = digital.ofdm_carrier_allocator_cvc(
            fft_len,
            occupied_carriers_all,
            pilot_carriers_all,
            pilot_symbols,
            (),                 # sync_words
            len_tag_key,
            True                # output_is_shifted
        )

        self.ifft = fft.fft_vcc(fft_len, False, (), True, 1)   # IFFT, shift=True
        self.cp_adder = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, len_tag_key)
        self.throttle_tx = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        _safe_set_min_output_buffer(self.chunks_to_symbols, (2 * packet_len * 4))
        _safe_set_min_output_buffer(self.carrier_alloc, (2 * transpose_len))
        _safe_set_min_output_buffer(self.cp_adder, int(2 * transpose_len * (fft_len + cp_len)))

        # --------------------------
        # Comms channel
        # --------------------------
        self.comm_chan = channels.channel_model(
            noise_voltage=comm_noise_voltage,
            frequency_offset=(comm_freq_offset_sc * 1.0 / fft_len),
            epsilon=1.0,
            taps=[1.0 + 0.0j],
            noise_seed=0,
            block_tags=False
        )

        self.comm_cp_rm = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, cp_len, len_tag_key)
        self.comm_fft = fft.fft_vcc(fft_len, True, (), True, 1)  # FFT, shift=True
        _safe_set_min_output_buffer(self.comm_cp_rm, (2 * transpose_len))

        # 等化：提取 -26..26（包含 QPSK data + BPSK pilots）
        self.comm_equalize = isac_equalize_extract_bins_vcvc(
            fft_len=fft_len,
            bins=occ_bins_all,
            out_vlen=n_total,
            len_tag_key=len_tag_key
        )

        self.comm_vec_to_stream = blocks.vector_to_stream(gr.sizeof_gr_complex, n_total)

        # 启动瞬态丢弃
        self.comm_skip = blocks.skiphead(gr.sizeof_gr_complex, n_total * 5)

        # 丢掉 padding 产生的 0 点（只用于星座显示）
        self.drop_small = drop_small_mag_cc(threshold=0.2)

        self.comm_const_sink = qtgui.const_sink_c(
            4096, "Constellation: QPSK(data) + BPSK(pilots) => 6 points", 1, None
        )
        self.comm_const_sink.set_update_time(0.10)
        self.comm_const_sink.enable_autoscale(True)
        self._comm_const_sink_win = sip.wrapinstance(self.comm_const_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._comm_const_sink_win)

        self.comm_freq_sink = qtgui.freq_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS, 0, samp_rate,
            "Comms Rx Spectrum (time-domain)", 1, None
        )
        self.comm_freq_sink.set_update_time(0.10)
        self.comm_freq_sink.set_y_axis(-140, 10)
        self.comm_freq_sink.enable_axis_labels(True)
        self._comm_freq_sink_win = sip.wrapinstance(self.comm_freq_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._comm_freq_sink_win)

        # --------------------------
        # Radar chain (buffers kept, transpose_len 已更新为 42)
        # --------------------------
        self.radar_target = radar.static_target_simulator_cc(
            [value_range], [velocity], [1e25], [0], [0],
            samp_rate, center_freq, -10, True, True, len_tag_key
        )
        _safe_set_min_output_buffer(self.radar_target, int(2 * transpose_len * (fft_len + cp_len)))

        self.radar_noise = analog.noise_source_c(analog.GR_GAUSSIAN, radar_noise_voltage, 0)
        self.radar_add = blocks.add_vcc(1)

        self.radar_cp_rm = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, cp_len, len_tag_key)
        self.radar_fft = fft.fft_vcc(fft_len, True, (), True, 1)
        _safe_set_min_output_buffer(self.radar_cp_rm, (2 * transpose_len))
        _safe_set_min_output_buffer(self.radar_fft, (2 * transpose_len))

        self.radar_div = radar.ofdm_divide_vcvc(
            fft_len, (fft_len * zeropadding_fac), (), 0, len_tag_key
        )
        _safe_set_min_output_buffer(self.radar_div, (2 * transpose_len))

        self.radar_range_fft = fft.fft_vcc(
            (fft_len * zeropadding_fac), True,
            window.blackmanharris(fft_len * zeropadding_fac),
            False, 1
        )
        _safe_set_min_output_buffer(self.radar_range_fft, (2 * transpose_len))

        self.radar_transpose_0 = radar.transpose_matrix_vcvc((fft_len * zeropadding_fac), transpose_len, len_tag_key)
        _safe_set_min_output_buffer(self.radar_transpose_0, (2 * fft_len * zeropadding_fac))

        self.radar_doppler_fft = fft.fft_vcc(transpose_len, False, window.blackmanharris(transpose_len), False, 1)
        _safe_set_min_output_buffer(self.radar_doppler_fft, (2 * transpose_len))

        self.radar_transpose_1 = radar.transpose_matrix_vcvc(transpose_len, (fft_len * zeropadding_fac), len_tag_key)
        _safe_set_min_output_buffer(self.radar_transpose_1, (2 * transpose_len))

        self.radar_mag2 = blocks.complex_to_mag_squared((fft_len * zeropadding_fac))
        self.radar_log = blocks.nlog10_ff(1, (fft_len * zeropadding_fac), 0)
        _safe_set_min_output_buffer(self.radar_mag2, (2 * transpose_len))
        _safe_set_min_output_buffer(self.radar_log, (2 * transpose_len))

        self.radar_plot = radar.qtgui_spectrogram_plot(
            (fft_len * zeropadding_fac),
            500,
            "range",
            "velocity",
            "OFDM Radar (ISAC)",
            [0, R_max],
            [0, v_max],
            [-15, -8],
            True,
            len_tag_key
        )
        _safe_set_min_output_buffer(self.radar_plot, (2 * transpose_len))

        self.radar_cfar = radar.os_cfar_2d_vc(
            (fft_len * zeropadding_fac),
            [10, 10],
            [0, 0],
            0.78,
            30,
            len_tag_key
        )
        self.radar_est = radar.estimator_ofdm(
            "range", (fft_len * zeropadding_fac), [0, R_max],
            "velocity", transpose_len,
            [0, v_max / 5 * 2.4, -v_max / 5 * 2.4, 0],
            False
        )
        self.radar_print = radar.print_results(False, "")

        # --------------------------
        # Connections
        # --------------------------
        self.connect(self.src_bytes, self.throttle_bytes, self.stream_to_tagged,
                     self.repack_8_to_sym, self.chunks_to_symbols, self.carrier_alloc,
                     self.ifft, self.cp_adder, self.throttle_tx)

        # Split to comm & radar
        self.connect((self.throttle_tx, 0), (self.comm_chan, 0))
        self.connect((self.throttle_tx, 0), (self.radar_target, 0))

        # Comms RX
        self.connect((self.comm_chan, 0), (self.comm_freq_sink, 0))
        self.connect((self.comm_chan, 0), (self.comm_cp_rm, 0))
        self.connect((self.comm_cp_rm, 0), (self.comm_fft, 0))

        # Equalize: TX = carrier_alloc (freq-domain), RX = comm_fft (freq-domain)
        self.connect((self.carrier_alloc, 0), (self.comm_equalize, 0))
        self.connect((self.comm_fft, 0), (self.comm_equalize, 1))

        self.connect((self.comm_equalize, 0), (self.comm_vec_to_stream, 0))
        self.connect((self.comm_vec_to_stream, 0), (self.comm_skip, 0))
        self.connect((self.comm_skip, 0), (self.drop_small, 0))
        self.connect((self.drop_small, 0), (self.comm_const_sink, 0))

        # Radar: target + noise -> add -> CP rm -> FFT -> divide -> RD
        self.connect((self.radar_noise, 0), (self.radar_add, 0))
        self.connect((self.radar_target, 0), (self.radar_add, 1))
        self.connect((self.radar_add, 0), (self.radar_cp_rm, 0))
        self.connect((self.radar_cp_rm, 0), (self.radar_fft, 0))

        self.connect((self.carrier_alloc, 0), (self.radar_div, 0))
        self.connect((self.radar_fft, 0), (self.radar_div, 1))

        self.connect((self.radar_div, 0), (self.radar_range_fft, 0))
        self.connect((self.radar_range_fft, 0), (self.radar_transpose_0, 0))
        self.connect((self.radar_transpose_0, 0), (self.radar_doppler_fft, 0))
        self.connect((self.radar_doppler_fft, 0), (self.radar_transpose_1, 0))
        self.connect((self.radar_transpose_1, 0), (self.radar_mag2, 0))
        self.connect((self.radar_mag2, 0), (self.radar_log, 0))
        self.connect((self.radar_log, 0), (self.radar_plot, 0))
        self.connect((self.radar_transpose_1, 0), (self.radar_cfar, 0))

        self.msg_connect((self.radar_cfar, "Msg out"), (self.radar_est, "Msg in"))
        self.msg_connect((self.radar_est, "Msg out"), (self.radar_print, "Msg in"))

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

    # sliders
    def set_value_range(self, value_range):
        self.value_range = float(value_range)
        self.radar_target.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0],
                                        self.samp_rate, self.center_freq, -10, True, True)

    def set_velocity(self, velocity):
        self.velocity = float(velocity)
        self.radar_target.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0],
                                        self.samp_rate, self.center_freq, -10, True, True)

    def set_comm_noise_voltage(self, nv):
        self.comm_noise_voltage = float(nv)
        self.comm_chan.set_noise_voltage(self.comm_noise_voltage)

    def set_comm_freq_offset_sc(self, fo_sc):
        self.comm_freq_offset_sc = float(fo_sc)
        self.comm_chan.set_frequency_offset(self.comm_freq_offset_sc * 1.0 / self.fft_len)

    def set_radar_noise_voltage(self, nv):
        self.radar_noise_voltage = float(nv)
        self.radar_noise.set_amplitude(self.radar_noise_voltage)


def main():
    qapp = SafeApplication(sys.argv)
    tb = ofdm_isac()
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
