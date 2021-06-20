def convertLabels():
    print("converting labels to 0 and 1"+"\n")
    # create 4 files for saving converted labels
    conv_labels_0 = open("data/labels_class_0.dat",'w')
    conv_labels_1 = open("data/labels_class_1.dat",'w')
    conv_labels_2 = open("data/labels_class_2.dat",'w')
    conv_labels_3 = open("data/labels_class_3.dat",'w')

    with open('data/labels_0.dat', 'r') as f:
        for val in f:
            if float(val) > 4.5:
                conv_labels_0.write(str(1) + "\n")
            else:
                conv_labels_0.write(str(0) + "\n")
    conv_labels_0.close()
    print(" Encoded label 0"+"\n")

    with open('data/labels_1.dat', 'r') as f:
        for val in f:
            if float(val) > 4.5:
                conv_labels_1.write(str(1) + "\n")
            else:
                conv_labels_1.write(str(0) + "\n")
    conv_labels_1.close()
    print(" Encoded label 1"+"\n")

    with open('data/labels_2.dat', 'r') as f:
        for val in f:
            if float(val) > 4.5:
                conv_labels_2.write(str(1) + "\n")
            else:
                conv_labels_2.write(str(0) + "\n")
    conv_labels_2.close()
    print(" Encoded label 2"+"\n")

    with open('data/labels_3.dat', 'r') as f:
        for val in f:
            if float(val) > 4.5:
                conv_labels_3.write(str(1) + "\n")
            else:
                conv_labels_3.write(str(0) + "\n")
    conv_labels_3.close()
    print(" Encoded label 3"+"\n")
if __name__=="__main__":
    convertLabels()

