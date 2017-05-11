# coding:utf-8


useinfo = open('H:/Datamining/rong360/credit/train/user_info_train.txt')
overdue = open('H:/Datamining/rong360/credit/train/overdue_train.txt')
user = []
for i in useinfo:
    line = i.replace("\n", "").split(",")
    linelist = [line[0], line[1], line[2], line[3], line[4], line[5]]
    for j in overdue:
        line2 = j.replace("\n", "").split(",")
        if line[0] == line2[0]:
            linelist.append(line2[1])
            break
    user.append(linelist)

# for i in overdue:
#     line = i.replace("\n", "").split(",")
#     linelist = [line[0]]
#     for j in useinfo:
#         line2 = j.replace("\n", "").split(",")
#         if line[0] == line2[0]:
#             linelist.append(line2[1])
#             linelist.append(line2[2])
#             linelist.append(line2[3])
#             linelist.append(line2[4])
#             linelist.append(line2[5])
#             linelist.append(line[1])
#     user.append(linelist)

print user[1]