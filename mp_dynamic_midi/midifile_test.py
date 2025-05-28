import mido
from mido import MidiFile

mid = MidiFile('../midi/2_0007_3_2472_0.mid')

outport = mido.open_output('IAC Driver Bus 1', autoreset=True)

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)

for msg in mid.play():
    outport.send(msg)