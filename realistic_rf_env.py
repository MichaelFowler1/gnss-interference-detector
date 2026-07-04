import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class RealisticRFEnvironment:
    """
    Generates synthetic I/Q data with high-fidelity channel impairments, 
    hardware constraints, and adversarial EW jamming profiles.
    """
    def __init__(self, sample_rate=1e6, num_symbols=1024, sps=8):
        self.fs = sample_rate
        self.num_symbols = num_symbols
        self.sps = sps
        self.num_samples = self.num_symbols * self.sps
        self.t = np.arange(self.num_samples) / self.fs
        self.signal_power = 1.0 

    def _rrc_filter(self, alpha=0.35, span=10):
        t = np.arange(-span/2, span/2 + 1/self.sps, 1/self.sps)
        h = np.zeros(len(t))
        for i, tc in enumerate(t):
            if tc == 0.0:
                h[i] = 1.0 - alpha + 4 * alpha / np.pi
            elif abs(tc) == 1 / (4 * alpha):
                h[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                h[i] = (np.sin(np.pi * tc * (1 - alpha)) + 
                        4 * alpha * tc * np.cos(np.pi * tc * (1 + alpha))) / \
                       (np.pi * tc * (1 - (4 * alpha * tc)**2))
        return h / np.sqrt(np.sum(h**2))

    def generate_base_qpsk(self):
        """Generates pristine, uncorrupted baseband QPSK symbols."""
        symbols = (np.random.choice([-1, 1], self.num_symbols) + 
                   1j * np.random.choice([-1, 1], self.num_symbols)) / np.sqrt(2)
        upsampled = np.zeros(self.num_samples, dtype=complex)
        upsampled[::self.sps] = symbols
        rrc_taps = self._rrc_filter()
        return np.convolve(upsampled, rrc_taps, mode='same')

    def apply_channel_effects(self, iq_data, cfo_hz=5000, phase_error_rad=0.1, snr_db=20):
        """
        Simulates the physical channel and hardware imperfections.
        - CFO/Doppler Shift (Frequency drift)
        - Phase Noise (Oscillator instability)
        - Multipath Fading (Reflections via Rayleigh fading)
        - Thermal Noise (AWGN)
        """
        # 1. Apply Carrier Frequency Offset (CFO) / Doppler Shift
        iq_cfo = iq_data * np.exp(1j * (2 * np.pi * cfo_hz * self.t + phase_error_rad))
        
        # 2. Apply Multipath Rayleigh Fading (3-path channel surrogate)
        # Simulates signal bouncing off structures or atmospheric layers
        channel_taps = np.array([1.0, 0.4*np.exp(1j*np.pi/4), 0.1*np.exp(1j*np.pi/2)])
        iq_faded = np.convolve(iq_cfo, channel_taps, mode='same')
        
        # 3. Add Ambient Thermal Noise (AWGN)
        noise_power = self.signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(iq_data)) + 
                                            1j * np.random.randn(len(iq_data)))
        
        return iq_faded + noise

    def apply_barrage_jamming(self, iq_data, jsr_db=15):
        """Injects high-power wideband noise."""
        jammer_power = self.signal_power * (10 ** (jsr_db / 10))
        jammer = np.sqrt(jammer_power / 2) * (np.random.randn(len(iq_data)) + 
                                              1j * np.random.randn(len(iq_data)))
        return iq_data + jammer

    def apply_tone_jamming(self, iq_data, jsr_db=15, offset_hz=150e3):
        """Injects narrow continuous-wave tone jamming."""
        jammer_power = self.signal_power * (10 ** (jsr_db / 10))
        jammer = np.sqrt(jammer_power) * np.exp(1j * 2 * np.pi * offset_hz * self.t)
        return iq_data + jammer

    def apply_hardware_clipping(self, iq_data, clip_percentile=98):
        """
        Models Receiver Frontend Saturation. High JSR attacks will saturate 
        the ADC. This creates severe non-linear distortion (harmonics).
        """
        # Calculate threshold based on the signal amplitude percentile
        amplitudes = np.abs(iq_data)
        threshold = np.percentile(amplitudes, clip_percentile)
        
        # Clip amplitudes but preserve the I/Q phase
        clipped_iq = np.copy(iq_data)
        mask = amplitudes > threshold
        clipped_iq[mask] = (threshold * (iq_data[mask] / amplitudes[mask]))
        return clipped_iq

# ==========================================
# Verification and Visual Inspection
# ==========================================
if __name__ == "__main__":
    env = RealisticRFEnvironment(sample_rate=1e6, num_symbols=2000)
    
    # 1. Generate clean baseband
    base_qpsk = env.generate_base_qpsk()
    
    # 2. Pass it through the atmosphere/hardware (Prone to Doppler, fading, thermal noise)
    contested_clear = env.apply_channel_effects(base_qpsk, cfo_hz=12000, snr_db=18)
    
    # 3. Simulate an EW attack scenario (Barrage jammer attempting to blind the receiver)
    jammed_signal = env.apply_barrage_jamming(contested_clear, jsr_db=20)
    
    # 4. Model the physical receiver maxing out due to the high-power jammer
    clipped_jammed_signal = env.apply_hardware_clipping(jammed_signal, clip_percentile=95)

    # Let's visualize the difference between the Contested Clear and the Severely Jammed + Clipped signal
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    for ax, (title, sig) in zip(axes, [("Operational Link (With Fading & Doppler Shift)", contested_clear), 
                                      ("Saturated Receiver under Electronic Attack (Barrage Jamming + ADC Clipping)", clipped_jammed_signal)]):
        f, Pxx = signal.welch(sig, env.fs, return_onesided=False, nperseg=1024)
        f = np.fft.fftshift(f) / 1e3
        Pxx = np.fft.fftshift(Pxx)
        
        ax.plot(f, 10 * np.log10(Pxx), color='teal' if 'Operational' in title else 'darkred')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel('PSD (dB/Hz)')
        
    axes[-1].set_xlabel('Frequency Offset (kHz)')
    plt.tight_layout()
    plt.show()