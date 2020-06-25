import pandas as pd

a = pd.read_csv("bounding_boxes.csv")
df = pd.DataFrame(data=a)

for rows in range(1,len(df)):
    row = pd.read_csv('bounding_boxes.csv', skiprows=rows, nrows=1, header=None)
    row = pd.DataFrame(data=row)

    frame_num = int(float(row[1]) + 1)

    width = (float(row[4]) - float(row[2]))
    height = (float(row[5]) - float(row[3]))

    width = width/800
    height = height/600
    
    x = float(row[4]) - width/2
    y = float(row[5]) - height/2

    x = x/800
    y = y/600
    
    f = open(f'./Frames/{str(frame_num).zfill(8)}.txt', 'a+')
    f.write(f'0 {x} {y} {width} {height} \n')
    f.close()