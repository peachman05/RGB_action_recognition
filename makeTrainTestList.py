# 18*3*10  a01_s10_e03

with open("trainlist02.txt", 'w') as the_file:
    for a in range(1,5): # 1 to 18
        for s in range(1,8): # 1 to 7
            for e in range(1,4): # 1 to 3
                action = "a"+"{:02d}".format(a)
                subject = "s"+"{:02d}".format(s)
                episode = "e"+"{:02d}".format(e)
                line = action+'/' + action+'_'+subject+'_'+episode #+'.mp4'
                # print(line)
                the_file.write(line+' '+str(a-1)+'\n')

with open('testlist02.txt', 'w') as the_file2:
    for a in range(1,5): # 1 to 18
        for s in range(8,11): # 8 to 10
            for e in range(1,4): # 1 to 3
                action = "a"+"{:02d}".format(a)
                subject = "s"+"{:02d}".format(s)
                episode = "e"+"{:02d}".format(e)
                line = action+'/' + action+'_'+subject+'_'+episode #+'.mp4'
                # print(line)
                the_file2.write(line+' '+str(a-1)+'\n')