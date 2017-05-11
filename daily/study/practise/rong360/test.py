# coding:utf-8

useinfo = open('H:/Datamining/rong360/credit/train/user_info_train.txt')
for line in useinfo:
    # linelist = line
    linelist = line.split(',')
    # print linelist
    print linelist[0], linelist[1], linelist[2], linelist[3], linelist[4], linelist[5]

# line = useinfo.readline()             # 调用文件的 readline()方法
# while line:
#     print line,                 # 后面跟 ',' 将忽略换行符
#     # print(line, end = '')　　　# 在 Python 3中使用
#     line = useinfo.readline()

useinfo.close()
a = [1,2,3]
print  a