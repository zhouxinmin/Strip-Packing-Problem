# coding:utf-8

import pandas as pd


# 特征工程
# 下面分别对user_info, bank_detail, browse_data, bill_detail, loan_data进行预处理

# user_info
# 读取数据集
user_info_train = pd.read_csv('H:/Datamining/rong360/credit/train/user_info_train.txt', header=None)
user_info_test = pd.read_csv('H:/Datamining/rong360/credit/test/user_info_test.txt', header=None)
# 设置字段（列）名称
col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
user_info_train.columns = col_names
user_info_test.columns = col_names
# 将userid（用户id）设置为数据集的index，并删除原userid所在列
user_info_train.index = user_info_train['userid']
user_info_train.drop('userid', axis=1, inplace=True)

user_info_test.index = user_info_test['userid']
user_info_test.drop('userid', axis=1, inplace=True)
print "user_info_test"
print user_info_test.head(5)


# 下面的处理方式类似，我仅注释不同的地方
# bank_detail
bank_detail_train = pd.read_csv('H:/Datamining/rong360/credit/train/bank_detail_train.txt', header=None)
bank_detail_test = pd.read_csv('H:/Datamining/rong360/credit/test/bank_detail_test.txt', header=None)
col_names = ['userid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
bank_detail_train.columns = col_names
bank_detail_test.columns = col_names
# 在该数据集中，一个用户对应多条记录，这里我们采用对每个用户每种交易类型取均值进行聚合
bank_detail_train_n = (bank_detail_train.loc[:, ['userid', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(['userid', 'trade_type']).mean()
bank_detail_test_n = (bank_detail_test.loc[:, ['userid', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(['userid', 'trade_type']).mean()
# 重塑数据集，并设置字段（列）名称
bank_detail_train_n = bank_detail_train_n.unstack()
bank_detail_train_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']

bank_detail_test_n = bank_detail_test_n.unstack()
bank_detail_test_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']
print "bank_detail_test_n"
print bank_detail_test_n.head(5)


# browse_history
browse_history_train = pd.read_csv('H:/Datamining/rong360/credit/train/browse_history_train.txt', header=None)
browse_history_test = pd.read_csv('H:/Datamining/rong360/credit/test/browse_history_test.txt', header=None)
col_names = ['userid', 'tm_encode_2', 'browse_data', 'browse_tag']
browse_history_train.columns = col_names
browse_history_test.columns = col_names
# browse_history = pd.concat([browse_history_train, browse_history_test])
# 这里采用计算每个用户总浏览行为次数进行聚合
browse_history_train_count = browse_history_train.loc[:, ['userid', 'browse_data']].groupby(['userid']).sum()
browse_history_test_count = browse_history_test.loc[:, ['userid', 'browse_data']].groupby(['userid']).sum()
print "browse_history_count"
print browse_history_test_count.head(5)


#
# bill_detail
bill_detail_train = pd.read_csv('H:/Datamining/rong360/credit/train/bill_detail_train.txt', header=None)
bill_detail_test = pd.read_csv('H:/Datamining/rong360/credit/test/bill_detail_test.txt', header=None)
col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']
bill_detail_train.columns = col_names
bill_detail_test.columns = col_names
# bill_detail = pd.concat([bill_detail_train, bill_detail_test])
bill_detail_train_mean = bill_detail_train.groupby(['userid']).mean()
bill_detail_train_mean.drop('bank_id', axis=1, inplace=True)

bill_detail_test_mean = bill_detail_test.groupby(['userid']).mean()
bill_detail_test_mean.drop('bank_id', axis=1, inplace=True)

print "bill_detail_mean"
print bill_detail_test_mean.head(5)

# loan_time
loan_time_train = pd.read_csv('H:/Datamining/rong360/credit/train/loan_time_train.txt', header=None)
loan_time_test = pd.read_csv('H:/Datamining/rong360/credit/test/loan_time_test.txt', header=None)
# loan_time = pd.concat([loan_time_train, loan_time_test])
loan_time_train.columns = ['userid', 'loan_time']
loan_time_train.index = loan_time_train['userid']

loan_time_test.columns = ['userid', 'loan_time']
loan_time_test.index = loan_time_test['userid']

loan_time_train.drop('userid', axis=1, inplace=True)
loan_time_test.drop('userid', axis=1, inplace=True)
print "loan_time"
print loan_time_test.head(5)


# 分别处理完以上数据集后，根据userid进行join，方式选择‘outer'，没有bill或者bank数据的user在对应字段上将为Na值
loan_train_data = user_info_train.join(bank_detail_train_n, how='outer')
loan_train_data = loan_train_data.join(browse_history_train_count, how='outer')
loan_train_data = loan_train_data.join(bill_detail_train_mean, how='outer')
loan_train_data = loan_train_data.join(loan_time_train, how='outer')

loan_test_data = user_info_test.join(bank_detail_test_n, how='outer')
loan_test_data = loan_test_data.join(browse_history_test_count, how='outer')
loan_test_data = loan_test_data.join(bill_detail_test_mean, how='outer')
loan_test_data = loan_test_data.join(loan_time_test, how='outer')

# # 填补缺失值
loan_train_data = loan_train_data.fillna(0.0)
print loan_train_data.head(5)

loan_test_data = loan_test_data.fillna(0.0)
print loan_test_data.head(5)

# 构造新特征（这里仅举个小例子）
loan_test_data['time'] = loan_test_data['loan_time'] - loan_test_data['tm_encode_3']
loan_train_data['time'] = loan_train_data['loan_time'] - loan_train_data['tm_encode_3']

# 对性别、职业等因子变量，构造其哑变量
category_col = ['sex', 'occupation', 'education', 'marriage', 'household']


def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col])
        dummy = dummy.add_prefix('{}#'.format(col))
        data.drop(col,
                  axis = 1,
                  inplace = True)
        data = data.join(dummy)
    return data
loan_test_data = set_dummies(loan_test_data, category_col)
loan_train_data = set_dummies(loan_train_data, category_col)

# overdue_train，这是我们模型所要拟合的目标
target = pd.read_csv('H:/Datamining/rong360/credit/train/overdue_train.txt', header=None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid', axis=1, inplace=True)

print len(loan_test_data)
print len(loan_train_data)
print len(target)
