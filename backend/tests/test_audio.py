import numpy as np

from morpheu import audio


def test_trim_silence_removes_leading_and_trailing_quiet():
    sr = 16000
    tone = 0.5 * np.ones(sr, dtype=np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    trimmed = audio.trim_silence(np.concatenate([silence, tone, silence]), sr)
    # keeps the tone plus at most the configured margin on each side
    assert sr <= len(trimmed) <= sr + 2 * int(0.1 * sr)


def test_trim_silence_keeps_all_silent_input_unchanged():
    silent = np.zeros(1000, dtype=np.float32)
    assert len(audio.trim_silence(silent, 16000)) == 1000


def test_normalize_peak_reaches_target():
    x = 0.1 * np.sin(np.linspace(0, 100, 10000)).astype(np.float32)
    y = audio.normalize_peak(x, target=0.9)
    assert abs(np.max(np.abs(y)) - 0.9) < 1e-3


def test_to_mono_float_averages_channels():
    stereo = np.stack([np.ones(10), -np.ones(10)], axis=1).astype(np.float32)
    assert np.allclose(audio.to_mono(stereo), 0.0)


def test_wav_roundtrip_reports_duration(tmp_path):
    sr = 24000
    path = tmp_path / "x.wav"
    audio.write_wav(path, np.zeros(sr * 2, dtype=np.float32), sr)
    data, read_sr = audio.read_audio(path)
    assert read_sr == sr
    assert abs(len(data) / read_sr - 2.0) < 1e-6
