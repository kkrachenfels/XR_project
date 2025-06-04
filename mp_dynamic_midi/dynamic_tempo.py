import mido
import time
import copy
import threading
import queue

from mido import MidiFile, tick2second

outport = mido.open_output('IAC Driver Bus 1', autoreset=True)


DEFAULT_VELOCITY = 64
PERCUSSION_CHANNEL = 4
KICK_NOTE = 36
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 500000
DEFAULT_TIME_SIGNATURE = {'numerator': 4, 'denominator': 4}

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

def kick_4_4():
    for i in range(4):
        if i % 4 == 0:
            velocity = DEFAULT_VELOCITY + 36
        else:
            velocity = DEFAULT_VELOCITY
        msg = mido.Message('note_on',
                        note=KICK_NOTE,
                        channel=PERCUSSION_CHANNEL,
                        velocity=velocity,
                        time=DEFAULT_TICKS_PER_BEAT/2
                        )
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_3_4():
    for i in range(3):
        if i % 3 == 0:
            velocity = DEFAULT_VELOCITY + 36
        else:
            velocity = DEFAULT_VELOCITY
        msg = mido.Message('note_on',
                        note=KICK_NOTE,
                        channel=PERCUSSION_CHANNEL,
                        velocity=velocity,
                        time=DEFAULT_TICKS_PER_BEAT/2
                        )
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_2_4():
    for i in range(4):
        if i % 2 == 0:
            velocity = DEFAULT_VELOCITY + 36
        else:
            velocity = DEFAULT_VELOCITY
        msg = mido.Message('note_on',
                        note=KICK_NOTE,
                        channel=PERCUSSION_CHANNEL,
                        velocity=velocity,
                        time=DEFAULT_TICKS_PER_BEAT/2
                        )
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)

def kick_all_beats():
    for i in range(4):
        velocity = DEFAULT_VELOCITY
        msg = mido.Message('note_on',
                        note=KICK_NOTE,
                        channel=PERCUSSION_CHANNEL,
                        velocity=velocity,
                        time=DEFAULT_TICKS_PER_BEAT/2
                        )
        time.sleep(convert_time_to_sec(msg.time))
        outport.send(msg)





# run a steady percussion beat in this thread
def percussion_thread(msg_q):
    running = True
    time_signature = copy.deepcopy(DEFAULT_TIME_SIGNATURE)
    current_percussion_fn = kick_4_4
    while running:
        try:
            # Non-blocking check for new commands
            command = msg_q.get_nowait()
            if 'numerator' in command.keys():
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


# Queue for communication between main thread and MIDI thread
command_queue = queue.Queue()

# Start MIDI thread
midi_thread = threading.Thread(target=percussion_thread, args=(command_queue,))
midi_thread.start()

# Main program example: Change behavior via queue
try:
    time.sleep(3)
    print("Changing time signature to 3/4")
    command_queue.put({'numerator': 3})

    time.sleep(3)
    print("Changing time signature to 2/4")
    command_queue.put({'numerator': 2})

    time.sleep(3)
    print("Upping tempo")
    command_queue.put({'tempo': 300000})

    time.sleep(3)
    print("Stopping MIDI thread")
    command_queue.put({'type': 'stop'})

    midi_thread.join()
    print("Thread terminated.")

except KeyboardInterrupt:
    print("Interrupted. Stopping...")
    command_queue.put({'type': 'stop'})
    midi_thread.join()


outport.close()   