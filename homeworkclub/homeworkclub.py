# coding:utf-8
import datetime


def counting(peoplenum):            # compute the num of gym
    T = peoplenum / 6
    X = peoplenum % 6
    if T == 0 and X < 4:
        return 0
    elif T == 0 and X >= 4:
        return 1
    elif T == 1:
        return 2
    elif T == 2 and X < 4:
        return 2
    elif T == 2 and X >= 4:
        return 3
    elif T == 3 and X < 4:
        return 3
    elif T == 3 and X >= 4:
        return 4
    else:
        return 4


def weeks(strtime):
    # 获取日期星期几,strtime 是字符串
    week = datetime.datetime.strptime(strtime, '%Y-%m-%d')
    return week.weekday() + 1


def costlist():        # 获取各小时的费用
    workday = []
    weekday = []
    for i in range(9, 22):
        if i < 12:
            workday.append(30)
            weekday.append(40)
        elif i < 18:
            workday.append(50)
            weekday.append(50)
        elif i < 20:
            workday.append(80)
            weekday.append(60)
        else:
            workday.append(60)
            weekday.append(60)

    return workday, weekday


def gymcost(time, worklist, weeklist, week):            # compute the cost of the day
    cost = 0
    starttime = int(time[:2])
    endtime = int(time[6:8])
    if week != 6 and week != 7:
        for t in range(starttime, endtime):
            cost += worklist[t - 9]
    else:
        for t in range(starttime, endtime):
            cost += weeklist[t - 9]
    return cost


def income(numpeo):            # compute the income of xiao ming
    if numpeo < 4:
        return 0
    else:
        return numpeo * 30


def pro(inputs):                # specification output
    timeser = inputs[:10]
    timezone = inputs[11:23]
    numpeo = int(inputs[23:])
    inCome = income(numpeo)
    gym = counting(numpeo)
    worklist, weeklist = costlist()
    week = weeks(timeser)
    totalCost = gym * gymcost(timezone, worklist, weeklist, week)
    profit = inCome - totalCost
    ab = inputs[:23]+' +'+str(inCome)+' -'+str(totalCost)
    if profit > 0:
        abd = ab+' +'
    elif profit == 0:
        abd = ab+' '
    else:
        abd = ab+' '
    abss = abd+str(profit)

    return abss+'\n', inCome, totalCost, profit


def disassembly(plan):            # 拆解字符串
    if plan[-1] == '\n':
        plan = plan[:-1]
    str2 = plan.split('\n')
    return str2


def summary(plan):
    outprint = '[Summary]\n\n'
    totalincome = 0
    totalpayment = 0
    Profit = 0
    prostr = disassembly(plan)
    for i in prostr:
        perday, income, totalcost, profit = pro(i)
        outprint += perday
        totalincome += income
        totalpayment += totalcost
        Profit += profit

    out =outprint + '\n' + 'Total Income:' + str(totalincome) + '\n' + 'Total Payment' + str(totalpayment) + '\n' \
         + 'Profit:'+ str(Profit)
    return out


if __name__ == '__main__':

    testInput = "2016-06-02 20:00~22:00 7\n2016-06-03 09:00~12:00 14\n2016-06-04 14:00~17:00 22\n" \
                "2016-06-05 19:00~22:00 3\n2016-06-06 12:00~15:00 15\n2016-06-07 15:00~17:00 12\n" \
                "2016-06-08 10:00~13:00 19\n2016-06-09 16:00~18:00 16\n2016-06-10 20:00~22:00 5\n2016-06-11 13:00~15:00 11\n"

    print summary(testInput)