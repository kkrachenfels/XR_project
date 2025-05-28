import mido
import time
import copy
import threading
import queue
import random

from mido import MidiFile, tick2second

outport = mido.open_output('IAC Driver Bus 1', autoreset=True)

MIN_NOTE = 21
MAX_NOTE = 108
OCTAVE = 12 # semitones
DEFAULT_VELOCITY = 64
STRING_CHANNEL = 0
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 500000
DEFAULT_TIME_SIGNATURE = {'numerator': 4, 'denominator': 4}
c_maj_chord = [60, 64, 67]

def convert_time_to_sec(time):
    # Convert message time from absolute time
    # in ticks to relative time in seconds.
    if time > 0:
        delta = tick2second(time, DEFAULT_TICKS_PER_BEAT, DEFAULT_TEMPO)
        print(delta)
    else:
        delta = 0
    return delta

def upper_inversion(chord):
    chord.sort()
    if chord[-1] + OCTAVE > MAX_NOTE:
        print("Chord can't be raised further...")
        return
    stop_old_note_msg = mido.Message('note_off',
                        note=chord[0],
                        channel=STRING_CHANNEL,
                        velocity=DEFAULT_VELOCITY,
                        time=0
                        )
    outport.send(stop_old_note_msg)

    # invert by adding an octave to the top chord note
    chord[0] += OCTAVE
    start_new_note_msg = mido.Message('note_on',
                        note=chord[0],
                        channel=STRING_CHANNEL,
                        velocity=DEFAULT_VELOCITY,
                        time=0
                        )
    outport.send(start_new_note_msg)


def lower_inversion(chord):
    chord.sort()
    if chord[-1] - OCTAVE < MIN_NOTE:
        print("Chord can't be lowered further...")
        return
    stop_old_note_msg = mido.Message('note_off',
                        note=chord[-1],
                        channel=STRING_CHANNEL,
                        velocity=DEFAULT_VELOCITY,
                        time=0
                        )
    outport.send(stop_old_note_msg)

    # invert by lowering an octave from the top chord note
    chord[-1] -= OCTAVE
    start_new_note_msg = mido.Message('note_on',
                        note=chord[-1],
                        channel=STRING_CHANNEL,
                        velocity=DEFAULT_VELOCITY,
                        time=0
                        )
    outport.send(start_new_note_msg) 

# neg number to go down
def shift_chord_semitones(chord, semitones=1):
    end_current_chord(chord)
    if semitones > 0 and (max(chord) + semitones > MAX_NOTE):
        print("Chord can't be raised further...")
        return
    elif semitones < 0 and (min(chord) - semitones < MIN_NOTE):
        print("Chord can't be lowered further...")
        return
    for i in range(len(chord)):
        chord[i] += semitones
    play_new_chord(chord)

def shift_minor_chord(chord):
    # find the major third interval, and make it a minor third
    chord.sort()
    full_chord = chord + [chord[0] + OCTAVE]
    for i in range(len(full_chord)-1):
        if full_chord[i] + 3 == full_chord[i+1]:
            stop_old_note_msg = mido.Message('note_off',
                                note=chord[i],
                                channel=STRING_CHANNEL,
                                velocity=DEFAULT_VELOCITY,
                                time=0
                                )
            outport.send(stop_old_note_msg)

            chord[i] -= 1
    
            start_new_note_msg = mido.Message('note_on',
                                note=chord[i],
                                channel=STRING_CHANNEL,
                                velocity=DEFAULT_VELOCITY,
                                time=0
                                )
            outport.send(start_new_note_msg)



def play_new_chord(chord):
    for midi_note in chord:
        msg = mido.Message('note_on',
                            note=midi_note,
                            channel=STRING_CHANNEL,
                            velocity=DEFAULT_VELOCITY,
                            time=0
                            )
        outport.send(msg)

def end_current_chord(chord):
    for midi_note in chord:
        msg = mido.Message('note_off',
                            note=midi_note,
                            channel=STRING_CHANNEL,
                            velocity=DEFAULT_VELOCITY,
                            time=0
                            )
        outport.send(msg)


# run a steady stream of string notes
def string_thread(msg_q):
    running = True
    current_chord = c_maj_chord
    play_new_chord(current_chord)
    while running:
        try:
            # Non-blocking check for new commands
            command = msg_q.get_nowait()
            if 'invert' in command.keys():
                if command['invert'] == 1:
                    upper_inversion(current_chord)
                elif command['invert'] == -1:
                    lower_inversion(current_chord)
            elif 'shift' in command.keys():
                shift_chord_semitones(current_chord, command['shift'])
            elif 'progression' in command.keys():
                if command['progression'] == 'minor':
                    shift_minor_chord(current_chord)
            elif command['type'] == 'stop':
                end_current_chord(current_chord)
                running = False
            print(current_chord)
        except queue.Empty:
            pass



# Queue for communication between main thread and MIDI thread
command_queue = queue.Queue()

# Start MIDI thread
midi_thread = threading.Thread(target=string_thread, args=(command_queue,))
midi_thread.start()

# Main program example: Change behavior via queue
try:
    time.sleep(2)
    print("Inverting chord upward six times in three seconds")
    for i in range(1):
        command_queue.put({'invert': 1})
        time.sleep(0.5)

    """
    print("Inverting chord downward twelve times in six seconds")
    for i in range(6):
        command_queue.put({'invert': -1})
        time.sleep(0.5)

    time.sleep(2)
    print("Randomly shifting the semitones of the chord up/down within -12 to 12")
    for i in range(6):
        random_semitones = random.randint(-12, 12)
        command_queue.put({'shift': random_semitones})
        time.sleep(0.5)
    """
    time.sleep(2)
    print("Make chord minor")
    command_queue.put({'progression': 'minor'})

    time.sleep(2)
    print("Stopping MIDI thread")
    command_queue.put({'type': 'stop'})
    time.sleep(1)

    midi_thread.join()
    print("Thread terminated.")

except KeyboardInterrupt:
    print("Interrupted. Stopping...")
    command_queue.put({'type': 'stop'})
    time.sleep(2)
    midi_thread.join()


outport.close()   