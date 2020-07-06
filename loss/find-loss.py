#FILL IN MY FILE NAME HERE
data_type = "att-unet-128"
#data_type = "noatt-unet-320"

rawdata_filename = data_type + ".txt"
trainloss_filename = "trainloss-" + data_type + ".csv"
validationloss_filename = "validationloss-" + data_type + ".csv"
validationacc_filename = "validationaccuracy-" + data_type + ".csv"

with open(trainloss_filename, "a+") as wf1:
    with open(rawdata_filename, 'r') as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("loss: ") and line.endswith("500)"):
                wf1.write(line.split()[1][0:7] + ",")
'''
with open(validationloss_filename, "a+") as wf2:
    with open(rawdata_filename, 'r') as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("validation loss:"):
                wf2.write(line.split()[2][0:7] + ",")
'''
with open(validationacc_filename, "a+") as wf3:
    with open(rawdata_filename, 'r') as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("validation average IoU:"):
                wf3.write(line.split()[3][0:7] + ",")


