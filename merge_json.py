# Read in the file
for x in range(6):
    print(x)
    filename_in='/fs/clip-scratch/tasnim/hay2/DPR/DPR/train/downloads/data/retriever/nq-train_'+str(x)+'.json'
    filename='/fs/clip-scratch/tasnim/hay2/DPR/DPR/train/downloads/data/retriever/nq-train_output_'+str(x)+'.json'
    with open(filename_in, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('}{', '},{')

    # Write the file out again
    with open(filename, 'w') as file:
        if x==0:
            file.write('[')
        file.write(filedata)
        if x<5:
            file.write(',')
        if x==5:
            file.write(']')

