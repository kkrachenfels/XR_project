import mido
import time
import random
import copy
import threading
import queue
import argparse

from mido import MidiFile, tick2second

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--replay-notes", action="store_true")

# Parse the arguments
args = parser.parse_args()

outport = mido.open_output('IAC Driver Bus 1', autoreset=True)

MIN_NOTE = 21
MAX_NOTE = 108
OCTAVE = 12 # semitones
DEFAULT_VELOCITY = 64
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 500000
DEFAULT_TIME_SIGNATURE = {'numerator': 4, 'denominator': 4}
CHANNELS = {"STRING": 0, 
            "PERCUSSION": 4
            }
KICK_NOTE = 36
c_maj_chord = [60, 64, 67]

tempo = DEFAULT_TEMPO

def convert_time_to_sec(time):
    # Convert message time from absolute time
    # in ticks to relative time in seconds.
    if time > 0:
        delta = tick2second(time, DEFAULT_TICKS_PER_BEAT, tempo)
        print(delta)
    else:
        delta = 0
    return delta

def adjust_tempo(new_tempo):
    global tempo
    tempo = new_tempo

def create_percussion_message(velocity):
    return mido.Message('note_on',
                    note=KICK_NOTE,
                    channel=CHANNELS['PERCUSSION'],
                    velocity=velocity,
                    time=DEFAULT_TICKS_PER_BEAT
                    )

def kick_4_4():
    print("In kick_4_4")
    for i in range(4):
        if i % 4 == 0: v = DEFAULT_VELOCITY + 36
        else: v = DEFAULT_VELOCITY

        msg = create_percussion_message(v)
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_3_4():
    print("In kick_3_4")
    for i in range(3):
        if i % 3 == 0: v = DEFAULT_VELOCITY + 36
        else: v = DEFAULT_VELOCITY

        msg = create_percussion_message(v)
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_2_4():
    print("In kick_2_4")
    for i in range(2):
        if i % 2 == 0: v = DEFAULT_VELOCITY + 36
        else: v = DEFAULT_VELOCITY

        msg = create_percussion_message(v)
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_all_beats():
    print("In kick_all")
    v = DEFAULT_VELOCITY
    msg = create_percussion_message(v)
    time.sleep(convert_time_to_sec(msg.time))
    outport.send(msg)


def create_string_on_message(note):
    return mido.Message('note_on',
                    note=note,
                    channel=CHANNELS['STRING'],
                    velocity=DEFAULT_VELOCITY,
                    time=0
                    )

def create_string_off_message(note):
    return mido.Message('note_off',
                    note=note,
                    channel=CHANNELS['STRING'],
                    velocity=DEFAULT_VELOCITY,
                    time=0
                    )



def upper_inversion(chord, replay_notes=False):
    chord.sort()
    if chord[-1] + OCTAVE > MAX_NOTE:
        print("Chord can't be raised further...")
        return
    
    if replay_notes:
        end_current_chord(chord)
        chord[0] += OCTAVE
        play_new_chord(chord)
        return
    
    stop_old_note_msg = create_string_off_message(chord[0])
    outport.send(stop_old_note_msg)

    # invert by adding an octave to the top chord note
    chord[0] += OCTAVE
    start_new_note_msg = create_string_on_message(chord[0])
    outport.send(start_new_note_msg)


def lower_inversion(chord, replay_notes=False):
    chord.sort()
    if chord[-1] - OCTAVE < MIN_NOTE:
        print("Chord can't be lowered further...")
        return
    
    if replay_notes:
        end_current_chord(chord)
        chord[-1] -= OCTAVE
        play_new_chord(chord)
        return
    
    stop_old_note_msg = create_string_off_message(chord[-1])
    outport.send(stop_old_note_msg)

    # invert by lowering an octave from the top chord note
    chord[-1] -= OCTAVE
    start_new_note_msg = create_string_off_message(chord[-1])
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

def shift_minor_chord(chord, replay_notes=False):
    # find the major third interval, and make it a minor third
    chord.sort()
    full_chord = chord + [chord[0] + OCTAVE]
    for i in range(len(full_chord)-1):
        if full_chord[i] + 3 == full_chord[i+1]:
            if replay_notes:
                end_current_chord(chord)
                chord[i] += 1
                play_new_chord(chord)
                return

            stop_old_note_msg = create_string_off_message(chord[i])
            outport.send(stop_old_note_msg)

            chord[i] -= 1
    
            start_new_note_msg = create_string_on_message(chord[i])
            outport.send(start_new_note_msg)



def play_new_chord(chord):
    for midi_note in chord:
        msg = create_string_on_message(midi_note)
        outport.send(msg)

def end_current_chord(chord):
    for midi_note in chord:
        msg = create_string_off_message(midi_note)
        outport.send(msg)




# run a steady percussion beat in this thread
def percussion_thread(msg_q, chord_q):
    running = True
    time_signature = copy.deepcopy(DEFAULT_TIME_SIGNATURE)
    current_percussion_fn = kick_4_4
    while running:
        try:
            # Non-blocking check for new commands
            command = msg_q.get_nowait()
            if list(command.keys())[0] in ['invert', 'shift', 'progression']:
                chord_q.put(command)
            elif 'numerator' in command.keys():
                time_signature['numerator'] = command['numerator']
                if time_signature['numerator'] == 4:
                    current_percussion_fn = kick_4_4
                elif time_signature['numerator'] == 3:
                    current_percussion_fn = kick_3_4
                elif time_signature['numerator'] == 2:
                    current_percussion_fn = kick_2_4
                else: current_percussion_fn = kick_all_beats
            elif 'tempo' in command.keys():
                adjust_tempo(command['tempo'])
            elif command['type'] == 'stop':
                running = False
                break
        except queue.Empty:
            pass

        current_percussion_fn()


# run a steady stream of string/synth notes
def chord_thread(msg_q):
    running = True
    current_chord = c_maj_chord
    play_new_chord(current_chord)
    while running:
        try:
            # Non-blocking check for new commands
            command = msg_q.get_nowait()
            if 'invert' in command.keys():
                if command['invert'] == 1:
                    upper_inversion(current_chord, replay_notes=args.replay_notes)
                elif command['invert'] == -1:
                    lower_inversion(current_chord, replay_notes=args.replay_notes)
            elif 'shift' in command.keys():
                shift_chord_semitones(current_chord, command['shift'])
            elif 'progression' in command.keys():
                if command['progression'] == 'minor':
                    shift_minor_chord(current_chord, replay_notes=args.replay_notes)
            elif command['type'] == 'stop':
                end_current_chord(current_chord)
                running = False
            print(current_chord)
        except queue.Empty:
            pass


# Queue for communication between main thread and MIDI percussion
command_queue = queue.Queue()

# Queue for commmunication to the chord thread
chord_queue = queue.Queue()

# Start MIDI threads
p_thread = threading.Thread(target=percussion_thread, args=(command_queue,chord_queue))
p_thread.start()

c_thread = threading.Thread(target=chord_thread, args=(chord_queue,))
c_thread.start()

# Main program example: Change behavior via queue
try:
    print("Inverting chord upward 2 times")
    for i in range(2):
        command_queue.put({'invert': 1})
    
    time.sleep(2)
    print("Changing time signature to 3/4")
    command_queue.put({'numerator': 3})

    print("Inverting chord downward 2 times")
    for i in range(2):
        command_queue.put({'invert': -1})

    time.sleep(3)
    print("Changing time signature to 2/4")
    command_queue.put({'numerator': 2})

    print("Randomly shifting the semitones of the chord up/down within -12 to 12")
    for i in range(4):
        random_semitones = random.randint(-12, 12)
        command_queue.put({'shift': random_semitones})

    time.sleep(3)
    print("Changing time signature to all ckicks")
    command_queue.put({'numerator': 1}) 

    print("Randomly shifting the semitones of the chord up/down within -12 to 12")
    for i in range(4):
        random_semitones = random.randint(-12, 12)
        command_queue.put({'shift': random_semitones})

    time.sleep(5)
    print("Upping tempo")
    command_queue.put({'tempo': 300000})

    print("Randomly shifting the semitones of the chord up/down within -12 to 12")
    for i in range(4):
        random_semitones = random.randint(-12, 12)
        command_queue.put({'shift': random_semitones})

    time.sleep(10)
    print("Stopping MIDI threads")
    command_queue.put({'type': 'stop'})
    chord_queue.put({'type': 'stop'})

    p_thread.join()
    c_thread.join()
    print("Threads terminated.")

except KeyboardInterrupt:
    print("Interrupted. Stopping...")
    command_queue.put({'type': 'stop'})
    chord_queue.put({'type': 'stop'})
    p_thread.join()
    c_thread.join()


outport.close()   