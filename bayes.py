import math

def parsing(data):
    total = [0, 0]
    data_attr = []
    data_prob = []
    count = len(data[0])
    for i in range(count):
        data_attr.append({})
        data_prob.append({})
    for row in data:
        for i in range(count):
            if row[i] not in data_attr[i]:
                data_attr[i][row[i]] = [0,0]
                data_prob[i][row[i]] = [0,0]
            data_attr[i][row[i]][int(float((row[-1])))] += 1
        total[int(float(row[-1]))] += 1
    for i in range(count):
        for (attr, counts) in data_attr[i].items():
            data_prob[i][attr][0] = counts[0] / total[0]
            data_prob[i][attr][1] = counts[1] / total[1]
    return [total, data_prob]
def classifier(train,test):
    total = train[0]
    prior = train[1]

    prediction = [math.log(total[0] / (total[0] + total[1])), math.log(total[1] / (total[0] + total[1]))]
    for i in range(len(test[0])):
        for j in range(2):
            if test[i] in prior[i]:
                if prior[i][test[i]][j] != 0:
                    prediction[j] += math.log(prior[i][test[i]][j])
    if prediction[1] > prediction[0] :return 1
    else:return 0




