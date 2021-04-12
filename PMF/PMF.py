import numpy as np

class PMF():
    #N个用户，M个item,形成一个NxM的评分矩阵
    #设D为潜在的特征个数；DxN表示用户的潜在特征矩阵；DxM为商品的潜在特征矩阵
    def train(self,train_set,validate_set,test_set,num_user_N,num_item_M,learning_rate,D,regularizer_u,regularizer_i,max_iteration):
        #在以0为均值，0.1为标准差的高斯分布中分别随机抽样出用户和item各自对应的潜在特征矩阵
        U = np.random.normal(0, 0.1, (num_user_N, D))   #用户 NxD 高斯分布矩阵
        V = np.random.normal(0, 0.1, (num_item_M, D))   #item MxD 高斯分布矩阵

        #使用模型在验证集上的表现情况来决定是否可以提早停止训练
        #验证集上的均方根误差如果连续3次大于之前迭代的结果，则可提早停止训练
        pre_validate_rmse = 100.0
        larger_max = 3
        larger_count = 0

        print('Start Training')
        for iteration in range(max_iteration):
            loss = 0.0
            #使用训练数据集中的数据开始训练
            for data in train_set:
                user = data[0]
                item = data[1]
                rating = data[2]
                #进行评分预测
                predict_rating = np.dot(U[user],V[item].T)
                #计算预测值与真实值之间的偏差
                error = rating-predict_rating
                #代价函数：平方和误差
                loss += error**2
                #进行参数更新
                U[user] += learning_rate*(error*V[item]-regularizer_u*U[user])
                V[item] += learning_rate*(error*U[user]-regularizer_i*V[item])
                #模型的优化目标/损失函数 即论文的公式(4)
                loss += regularizer_u*np.square(U[user]).sum()+regularizer_i*np.square(V[item]).sum()
            loss = 0.5*loss

            #使用测试集进行模型性能的评估
            validate_rmse = self.evaluation_rmse(U, V, validate_set)
            test_rmse = self.evaluation_rmse(U,V,test_set)
            print('iteration:%d training loss:%.3f validate_rmse:%.5f test_rmse:%.5f'%(iteration,loss,validate_rmse,test_rmse))

            #验证集上的均方根误差如果连续3次大于之前迭代的结果，则可提早停止训练
            if validate_rmse < pre_validate_rmse:
                pre_validate_rmse = validate_rmse
                larger_count = 0
            else:
                larger_count += 1
            if larger_count >= larger_max:
                break

    #评价指标：root mean squared error 均方根误差/标准误差
    def evaluation_rmse(self,U,V,test):
        test_count = len(test)
        tmp_rmse = 0.0
        for test_data in test:
            user = test_data[0]
            item = test_data[1]
            real_rating = test_data[2]
            predict_rating = np.dot(U[user],V[item].T)
            tmp_rmse += np.square(real_rating-predict_rating)
        rmse = np.sqrt(tmp_rmse/test_count)
        return rmse

#读取数据，划分数据集
def read_data(path,train_ratio,validate_ratio,test_ratio):
    user_set = {}
    item_set = {}
    N_user = 0
    M_item = 0
    number_of_rating = 0
    data = []
    with open(path) as f:
        for line in f.readlines():
            #数据集：ml-latest-small/ratings.csv
            u,i,r,_ = line.split(',')
            #数据集：ml-100k/u.data
            #u,i,r,_=line.split('::')
            number_of_rating += 1
            if u not in user_set:
                user_set[u] = N_user
                N_user += 1
            if i not in item_set:
                item_set[i] = M_item
                M_item += 1
            data.append([user_set[u],item_set[i],float(r)])

    np.random.shuffle(data)
    train = data[0:int(len(data)*train_ratio)]
    validate = data[int(len(data)*train_ratio):int(len(data)*(train_ratio+validate_ratio))]
    test = data[int(len(data)*(train_ratio+validate_ratio)):]
    print('number of users:%d number of items:%d number of ratings:%d' % (N_user, M_item, number_of_rating))
    return train,validate,test,N_user,M_item

if __name__=='__main__':
    train_set,validate_set,test_set,num_user, num_item = read_data('C:/Users/Yaxuan/PycharmProjects/RecommenderSystem/Data/ml-latest-small/ratings.csv', 0.8,0.1,0.1)
    #其他版本的数据集 ml-100k
    # num_user, num_item, train, test = read_data('C:/Users/Yaxuan/PycharmProjects/RecommenderSystem/Data/Data/ml-100k/u.data', 0.8)
    pmf=PMF()
    pmf.train(train_set,validate_set,test_set,num_user,num_item,0.01,10,0.01,0.01,50)
    print('End')