matrix = []

readpath=open('filename.txt','r')
for line in readpath.readlines():
    num = line.strip('\n').split(' ')    ###filename.txt中内容如下
    numb = [int(x) for x in num]         ###5 6 7 8
    matrix.append(numb)                  ###1 2 3 4
matrix.sort(key = lambda x:x[:][0])      ###9 10 11 12
matrix[1],matrix[2]=matrix[2],matrix[1]  ###13 14 15 16
for line in range(0,len(matrix)):
    print matrix[line]

readpath.close()