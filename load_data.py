import pickle

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
no_of_users=32


def convertData(): 
    print("Loading Data Files of each subject." + "\n")
    # create 1 file for storing data and 4 files for storing labels
    file_data = open("data/features.dat",'w')
    file_labels0 = open("data/labels_0.dat",'w')
    file_labels1 = open("data/labels_1.dat",'w')
    file_labels2 = open("data/labels_2.dat",'w')
    file_labels3 = open("data/labels_3.dat",'w')

    # For each user, open data file using pickle 
    for i in range(no_of_users): 
        #check if i is integer 
        if(i%1==0):
            # deciding file names
            if i<10:
                name = '%0*d' % (2,i+1) #for assigning names as 01,02 etc
            else:
                name = i+1

        # complete file name
        fname = 'data/s'+str(name)+'.dat'

        # read file using pickle module in Binary mode
        data = pickle.load(open(fname,'rb'), encoding='latin1')    
        print(fname)

    # Extracting labels from data
        # for each trial, extract label from data
        for tr in range(nTrial):
            if(tr%1==0):
                for t in range(nTime):
                # downsampling data by taking only 32th milisecond's data 
                    if t%32 == 0:
                        for ch in range(nChannel):
                            # storing channel number with that channel's data
                            file_data.write(str(ch+1) + " ");
                            file_data.write(str(data['data'][tr][ch][t]) + " ");

                #saving label of each trial
                file_labels0.write(str(data['labels'][tr][0]) + "\n");
                file_labels1.write(str(data['labels'][tr][1]) + "\n");
                file_labels2.write(str(data['labels'][tr][2]) + "\n");
                file_labels3.write(str(data['labels'][tr][3]) + "\n");

            file_data.write("\n");
                    

    # Close all files
    file_labels0.close()
    file_labels1.close()
    file_labels2.close()
    file_labels3.close()
    file_data.close()
    print("\n" + "Loaded and Downsampled all data files successfully.")

if __name__ == '__main__':
    convertData()
