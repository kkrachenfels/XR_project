# Collaborative Dance to Music: Real-Time Dynamic Music Creation via Multi-person Movement Tracking

## Running the MIDI-based thread
- Install any necessary packages required to run the code beforehand; refer to `mediapipe_midi.yml` 
- Open a digital audio workstation (DAW) of choice and create three virtual instrument channels, with two supporting standard piano notes and one being a drum kit. Our system runs on MacOS and works by default with Garageband/Logic Pro; if you are using a different system you may need to select a different audio driver (e.g. change the setting for the MIDI output bus from `'IAC Driver Bus 1'` to your output bus on line 17 of `midi_thread.py`). 
- Run `python laban_based_tracking.py` to boot up the main script and webcam. This will also boot up the midi thread on startup.
- Have fun!

## Running the GenAI-based thread
