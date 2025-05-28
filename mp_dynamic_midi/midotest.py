import mido
import time

print(mido.backend)

outport = mido.open_output('IAC Driver Bus 1', autoreset=True)

c_maj_chord = [24, 28, 31]

DEFAULT_VELOCITY = 80

while c_maj_chord[0] < 108:
    for channel, note in enumerate(c_maj_chord):
        outport.send(mido.Message('note_on', channel=channel, note=note, velocity=DEFAULT_VELOCITY))
    time.sleep(2)
    for channel, note in enumerate(c_maj_chord):
        outport.send(mido.Message('note_off', channel=channel, note=note, velocity=DEFAULT_VELOCITY)) 
    
    # go up an octave
    c_maj_chord = [n+12 for n in c_maj_chord]

outport.close()        


'''

c_maj_notes = [60, 64, 67]

on_msgs = []
off_msgs = []
for channel, note in enumerate(c_maj_notes):
    on_msgs.append(mido.Message('note_on', channel=channel, note=note, velocity=80))
    off_msgs.append(mido.Message('note_off', channel=channel, note=note, velocity=80))

for msg in on_msgs:
    print(msg.dict())
    outport.send(msg)

time.sleep(3)

for i, msg in enumerate(off_msgs):
    outport.send(msg)
    on_msgs[i].note -= 12
    outport.send(on_msgs[i])

time.sleep(3)

for channel in range(16):
    for note in range(128):
        outport.send(mido.Message('note_off', note=note, velocity=0, channel=channel))

outport.close()

# or kill all notes
# outport.reset()
# outport.panic()

'''
