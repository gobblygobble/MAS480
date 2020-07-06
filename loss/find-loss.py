import os
'''
# open all files in current directory
for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        # work with files that end with '.txt'
        if not filename.endswith(".txt"):
            continue
        # note that f.name will return filename!
        print("NEW FILE: {}".format(filename))
        for line in f:
            print(line.rstrip())
'''
with open("csv-unet-without-res-64.csv", "a+") as wf:
    with open("unet-64-without-res-loss.txt", 'r') as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("loss: ") and line.endswith("500)"):
                wf.write(line.split()[1][0:7] + ",")
