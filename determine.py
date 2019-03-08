import bayes
import dtree_build
import math
import csv
from sklearn.utils import resample

train = []
test = []
col = ['ZIP_CODE', 'TOTAL_VISITS', 'TOTAL_SPENT', 'HAS_CREDIT_CARD', 'AVRG_SPENT_PER_VISIT', 'PSWEATERS', 'PKNIT_TOPS',
       'PKNIT_DRES', 'PBLOUSES', 'PJACKETS', 'PCAR_PNTS', 'PCAS_PNTS', 'PSHIRTS', 'PDRESSES', 'PSUITS', 'POUTERWEAR',
       'PJEWELRY', 'PFASHION', 'PLEGWEAR', 'PCOLLSPND', 'AMSPEND', 'PSSPEND', 'CCSPEND', 'AXSPEND', 'SPEND_LAST_MONTH',
       'SPEND_LAST_3MONTH', 'SPEND_LAST_6MONTH', 'SPENT_LAST_YEAR', 'GMP', 'PROMOS_ON_FILE', 'DAYS_ON_FILE',
       'FREQ_DAYS', 'MARKDOWN', 'PRODUCT_CLASSES', 'COUPONS', 'STYLES', 'STORES', 'STORELOY', 'VALPHON', 'WEB',
       'MAILED', 'RESPONDED', 'RESPONSERATE', 'LTFREDAY', 'CLUSTYPE', 'PERCRET']

csv_file_name = "Arranged_Dataset.csv"
data = []
with open(csv_file_name) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data.append(list(row))

train = resample(data[1:], replace=True, n_samples=int(len(data)))
for i in data[1:]:
    if i not in train:
        test.append(i)

tree = dtree_build.buildtree(train, min_gain=0.01, min_samples=5)
dtree_build.printtree(tree, " ", col)
naive = bayes.parsing(train)
dcorrect = 0
ncorrect = 0
output = [['inst/#', 'actual', 'predicted', 'probability']]
count = 0
for item in test:
    count += 1
    input = bayes.classifier(naive, item)
    if input == int(float(item[-1])):
        ncorrect += 1
    result = dtree_build.classify(item, tree)
    total = 0
    max = 0
    str = ''
    for i, j in result.items():
        total += j
        if j >= max:
            max = j
            str = i
    if str == item[-1]:
        dcorrect += 1
    output.append([count, float(item[-1]), str, max / total])
with open("predicted.csv", "w") as new_csv:
    writer = csv.writer(new_csv)
    writer.writerows(output)
    print("Accuracy for dtree:" , dcorrect / len(test))
    print("Accuracy for nbayes:" , ncorrect / len(test))
