import mido
import time
import random
import copy
import threading
import queue
import argparse

from mido import MidiFile, tick2second, tempo2bpm

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
            "MELODY": 1,
            "PERCUSSION": 4
            }
KICK_NOTE = 36
SNARE = 34
HI_HAT_CLOSE = 33
c_maj_chord = [60, 64, 67]
c_maj_notes = [60, 62, 64, 65, 67, 69, 71, 72]
c_min_notes = [60, 62, 63, 65, 67, 68, 70, 72]
major_intervals = [2,4,5,7,9,11,12]
minor_intervals = [2,3,5,7,8,10,12]

tempo = DEFAULT_TEMPO

def convert_time_to_sec(time):
    # Convert message time from absolute time
    # in ticks to relative time in seconds.
    if time > 0:
        delta = tick2second(time, DEFAULT_TICKS_PER_BEAT, tempo)
        #print(delta)
    else:
        delta = 0
    return delta


def can_speedup(speedup):
    global tempo
    bpm = tempo2bpm(tempo) / 2
    print(bpm)
    new_bpm = bpm / speedup
    if new_bpm < 40 or new_bpm > 180:
        print("Can't adjust tempo further!!")
        return False
    return True

def adjust_tempo(speedup):
    global tempo
    tempo /= speedup # speedup is inverse bc of the way midi calculations are
    # so 0.8 speedup is kinda like 1.2 speedup 

def create_percussion_message(note=KICK_NOTE, velocity=DEFAULT_VELOCITY):
    return mido.Message('note_on',
                    note=note,
                    channel=CHANNELS['PERCUSSION'],
                    velocity=velocity,
                    time=DEFAULT_TICKS_PER_BEAT
                    )

def create_percussion_off_message(note=KICK_NOTE, velocity=DEFAULT_VELOCITY):
    return mido.Message('note_off',
                    note=note,
                    channel=CHANNELS['PERCUSSION'],
                    velocity=velocity,
                    time=DEFAULT_TICKS_PER_BEAT
                    )

def create_string_on_message(note, time=0):
    return mido.Message('note_on',
                    note=note,
                    channel=CHANNELS['STRING'],
                    velocity=DEFAULT_VELOCITY,
                    time=time
                    )

def create_string_off_message(note, time=0):
    return mido.Message('note_off',
                    note=note,
                    channel=CHANNELS['STRING'],
                    velocity=DEFAULT_VELOCITY,
                    time=time
                    )

def create_melody_on_message(note, time=0):
    return mido.Message('note_on',
                    note=note,
                    channel=CHANNELS['MELODY'],
                    velocity=DEFAULT_VELOCITY,
                    time=time
                    )

def create_melody_off_message(note, time=0):
    return mido.Message('note_off',
                    note=note,
                    channel=CHANNELS['MELODY'],
                    velocity=DEFAULT_VELOCITY,
                    time=time
                    )

def create_scale(bass_note, minor=False):
    scale = [bass_note]
    intervals = major_intervals
    if minor:
        intervals = minor_intervals
    for interval in intervals:
        scale.append(bass_note + interval)
    return scale


def play_new_chord(chord):
    for midi_note in chord:
        msg = create_string_on_message(midi_note)
        outport.send(msg)

def end_current_chord(chord):
    for midi_note in chord:
        msg = create_string_off_message(midi_note)
        outport.send(msg)


def time_4_4(chord, melody=None, shift=0, inversion=0, minor=False, replay_notes=False):
    print("In time_4_4")
    if not melody:
        melody = []
        scale = create_scale(chord[0], minor=minor)
        for i in range(7):
            melody.append(scale[random.randint(0, 7)]+OCTAVE)
    if melody and shift:
        print(shift)
        for i in range(len(melody)):
            melody[i] += shift

    outport.send(create_percussion_message(note=KICK_NOTE))
    shift_chord_semitones(chord, semitones=shift)
    if minor:
        shift_minor_chord(chord, replay_notes=replay_notes)
    else:
        shift_major_chord(chord, replay_notes=replay_notes)
    print(chord)
    print(melody)
    bass = chord[random.randint(0,2)] - OCTAVE
    outport.send(create_string_on_message(bass))
    outport.send(create_percussion_off_message(note=KICK_NOTE))

    for i in range(7):
        msg = create_melody_on_message(melody[i], time=DEFAULT_TICKS_PER_BEAT)
        sleep_time = convert_time_to_sec(msg.time)
        time.sleep(sleep_time)
        outport.send(msg) 
        outport.send(create_percussion_message(note=HI_HAT_CLOSE))
        if i == 3:
            outport.send(create_percussion_message(note=SNARE))
            outport.send(create_percussion_off_message(note=SNARE))
        if i != 0:
            msg = create_melody_off_message(melody[i-1])
            outport.send(msg)  

    time.sleep(convert_time_to_sec(DEFAULT_TICKS_PER_BEAT))
    outport.send(create_melody_off_message(melody[-1]))
    outport.send(create_string_off_message(bass))
    return melody


def time_3_4(chord, melody=None, shift=0, inversion=0, minor=False, replay_notes=False):
    print("In time_4_4")
    if not melody:
        melody = []
        scale = create_scale(chord[0], minor=minor)
        for i in range(7):
            melody.append(scale[random.randint(0, 7)]+OCTAVE)
    if melody and shift:
        print(shift)
        for i in range(len(melody)):
            melody[i] += shift

    outport.send(create_percussion_message(note=KICK_NOTE))
    shift_chord_semitones(chord, semitones=shift)
    if minor:
        shift_minor_chord(chord, replay_notes=replay_notes)
    else:
        shift_major_chord(chord, replay_notes=replay_notes)
    print(chord)
    print(melody)
    bass = chord[random.randint(0,2)] - OCTAVE
    outport.send(create_string_on_message(bass))
    outport.send(create_percussion_off_message(note=KICK_NOTE))

    for i in range(5):
        msg = create_melody_on_message(melody[i], time=DEFAULT_TICKS_PER_BEAT)
        sleep_time = convert_time_to_sec(msg.time)
        time.sleep(sleep_time)
        outport.send(msg) 
        outport.send(create_percussion_message(note=HI_HAT_CLOSE))
        if i == 2:
            outport.send(create_percussion_message(note=SNARE))
            outport.send(create_percussion_off_message(note=SNARE))
        if i != 0:
            msg = create_melody_off_message(melody[i-1])
            outport.send(msg)  

    time.sleep(convert_time_to_sec(DEFAULT_TICKS_PER_BEAT))
    outport.send(create_melody_off_message(melody[-1]))
    outport.send(create_string_off_message(bass))
    return melody


def time_2_4(chord, melody=None, shift=0, inversion=0, minor=False, replay_notes=False):
    print("In time_4_4")
    if not melody:
        melody = []
        scale = create_scale(chord[0], minor=minor)
        for i in range(7):
            melody.append(scale[random.randint(0, 7)]+OCTAVE)
    if melody and shift:
        print(shift)
        for i in range(len(melody)):
            melody[i] += shift

    outport.send(create_percussion_message(note=KICK_NOTE))
    shift_chord_semitones(chord, semitones=shift)
    if minor:
        shift_minor_chord(chord, replay_notes=replay_notes)
    else:
        shift_major_chord(chord, replay_notes=replay_notes)
    print(chord)
    print(melody)
    bass = chord[random.randint(0,2)] - OCTAVE
    outport.send(create_string_on_message(bass))
    outport.send(create_percussion_off_message(note=KICK_NOTE))

    for i in range(3):
        msg = create_melody_on_message(melody[i], time=DEFAULT_TICKS_PER_BEAT)
        sleep_time = convert_time_to_sec(msg.time)
        time.sleep(sleep_time)
        outport.send(msg) 
        outport.send(create_percussion_message(note=HI_HAT_CLOSE))
        if i == 2:
            outport.send(create_percussion_message(note=SNARE))
            outport.send(create_percussion_off_message(note=SNARE))
        if i != 0:
            msg = create_melody_off_message(melody[i-1])
            outport.send(msg)  

    time.sleep(convert_time_to_sec(DEFAULT_TICKS_PER_BEAT))
    outport.send(create_melody_off_message(melody[-1]))
    outport.send(create_string_off_message(bass))
    return melody


# neg number to go down
def shift_chord_semitones(chord, semitones=1):
    end_current_chord(chord)
    if semitones > 0 and (max(chord) + semitones > (MAX_NOTE - OCTAVE)):
        print("Chord can't be raised further...")
        return
    elif semitones < 0 and (min(chord) - semitones < MIN_NOTE):
        print("Chord can't be lowered further...")
        return
    for i in range(len(chord)):
        chord[i] += semitones
    play_new_chord(chord)


def shift_major_chord(chord, replay_notes=False):
    # already a major chord
    if chord[1] == chord[0] + 4:
        return
    
    else:
        if replay_notes:
            end_current_chord(chord)
            chord[1] += 1
            play_new_chord(chord)
            return
        else:
            stop_old_note_msg = create_string_off_message(chord[1])
            outport.send(stop_old_note_msg)

            chord[1] += 1
    
            start_new_note_msg = create_string_on_message(chord[1])
            outport.send(start_new_note_msg)

def shift_minor_chord(chord, replay_notes=False):
    # already a minor chord
    if chord[1] == chord[0] + 3:
        return
    
    else:
        if replay_notes:
            end_current_chord(chord)
            chord[1] -= 1
            play_new_chord(chord)
            return
        else:
            stop_old_note_msg = create_string_off_message(chord[1])
            outport.send(stop_old_note_msg)

            chord[1] -= 1
    
            start_new_note_msg = create_string_on_message(chord[1])
            outport.send(start_new_note_msg)



def end_all_notes():
    for note in range(MIN_NOTE, MAX_NOTE+1):
        outport.send(create_string_off_message(note))
        outport.send(create_melody_off_message(note))

# sometimes percussion MIDI dies after a while, so refresh it
def reset_percussion():
    outport.send(create_percussion_off_message(HI_HAT_CLOSE))
    outport.send(create_percussion_off_message(KICK_NOTE))
    outport.send(create_percussion_off_message(SNARE))


# run a steady stream of string/synth notes
def music_thread_v2(msg_q, return_q=None):
    running = True
    current_chord = c_maj_chord
    melody = None
    shift = 0
    current_fn = time_4_4
    minor = False
    while running:
        shift = 0
        command = None
        try:
            # Non-blocking check for new commands
            while command := msg_q.get_nowait():
                print(f"====>{command}")
                if 'type' in command.keys():
                    if command['type'] == 'stop':
                        end_all_notes()
                        running = False
                        break
                    elif command['type'] == 'new_melody':
                        melody = []
                elif 'tempo' in command.keys():
                    perform_speedup = can_speedup(command['tempo'])
                    if perform_speedup:
                        adjust_tempo(command['tempo'])
                        reset_percussion()
                    else:
                        continue
                elif 'shift' in command.keys():
                    shift += command['shift']
                elif 'progression' in command.keys():
                    end_all_notes()
                    if command['progression'] == 'minor':
                        minor=True
                    else:
                        minor=False
                    melody = []
                elif 'time' in command.keys():
                    if command['time'] == 4:
                        current_fn = time_4_4
                    elif command['time'] == 3:
                        current_fn = time_3_4
                    elif command['time'] == 2:
                        current_fn = time_2_4
                    reset_percussion()

                # notify main thread that we processed this command
                return_q.put(command)
        except queue.Empty:
            pass

        melody = current_fn(current_chord, melody=melody, shift=shift, minor=minor)




# test midi thread commands by itself before using mediapipe
if __name__ == "__ main __":
    # Queue for communication between main thread and MIDI music
    command_queue = queue.Queue()

    # Start MIDI thread
    m_thread = threading.Thread(target=music_thread_v2, args=(command_queue,))
    m_thread.start()

    try:
        #time.sleep(1)
        print("Upping tempo")
        command_queue.put({'tempo': 300000})

        #command_queue.put({'type': 'new_melody'})

        #command_queue.put({'invert': 1})
        #command_queue.put({'invert': -1})

        print("Randomly shifting the semitones of the chord up/down within -12 to 12")
        for i in range(1):
            random_semitones = random.randint(-12, 12)
            command_queue.put({'shift': random_semitones})
            #command_queue.put({'type': 'new_melody'}) 
        command_queue.put({'progression': 'minor'})
        for i in range(1):
            random_semitones = random.randint(-12, 12)
            command_queue.put({'shift': random_semitones})

        command_queue.put({'progression': 'major'})

        print("changing time????")
        command_queue.put({'time': 3})
        time.sleep(3)
        #command_queue.put({'time': 2})


        #time.sleep(10)
        #print("Stopping MIDI threads")
        #command_queue.put({'type': 'stop'})
        #chord_queue.put({'type': 'stop'})

        m_thread.join()
        print("Threads terminated.")

    except KeyboardInterrupt:
        print("Interrupted. Stopping...")
        command_queue.put({'type': 'stop'})
        m_thread.join()

    outport.close()   
