import csv


csv_file = csv.reader(open('/Users/zhaozehua/Desktop/ali music/mars_tianchi_user_actions.csv','r'))

training = open('/Users/zhaozehua/Desktop/ali music/training.csv','w')
csv_write_train = csv.writer(training,dialect='excel')
testing = open('/Users/zhaozehua/Desktop/ali music/testing.csv','w')
csv_write_test = csv.writer(testing,dialect='excel')

for line in csv_file:
    if line[4] <= '20150831' and line[4] >= '20150701':
        csv_write_test.writerow(line)
    if line[4] <= '20150630' and line[4] >= '20150301':
        csv_write_train.writerow(line)

