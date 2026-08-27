# audio — acoustic drone detection (Phase 3)

Not started. Gazebo does not simulate acoustics, so this component will
synthesize/replay microphone audio driven by simulator state (drone distance,
velocity, rotor regime) relative to the station's `mic_array_link` pose, then
run the spectrogram → features → detection pipeline on it.
