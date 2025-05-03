import pandas as pd


column_headers = []
for i in range(33):
    landmark = i+1
    column_headers += [f"{landmark}x", f"{landmark}y", f"{landmark}z"]
    #names=[str(i) for i in (range(33*3))]

def concat_csvs(note_name):
    names = ['kay', 'kaitlyn', 'daniel']
    csvs_to_concat = []
    for name in names:
        csvs_to_concat.append(pd.read_csv(f"data/{note_name}_{name}.csv", names=column_headers))
    concatted_csv = pd.concat(csvs_to_concat, axis=0)
    concatted_csv.to_csv(f"data/pose_data/{note_name}.csv", index=False)



for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    concat_csvs(note)
    concat_csvs(f"{note}_sharp")
    concat_csvs(f"{note}_flat")
