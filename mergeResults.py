import csv
import os

def read_result(path, file):

    try:
        csvfile = open(os.path.join(path, 'csvfiles', file))

    except IOError:
        print('cannot open')
        return -1

    my_content = csv.reader(csvfile, delimiter=',')
    csv_content_list = []
        
    for i, row in enumerate(my_content):
        if (i>1):
            csv_content_list.append(row)

    csvfile.close

    return csv_content_list

def create_csv(path, name_csvfile):

    if not os.path.exists(os.path.join(path, 'csvfiles', 'results')):
        os.makedirs(os.path.join(path, 'csvfiles', 'results'))
        
    if not os.path.exists(os.path.join(path, 'csvfiles', 'results', name_csvfile + '.csv')):
        with open(os.path.join(path, 'csvfiles', 'results', name_csvfile + '.csv'), 'w', newline='') as csvf:
            writer = csv.writer(csvf, delimiter=',')
            writer.writerow(['sep=,'])
            writer.writerow(['Name', 'Classifier', 'Parameter', 'Precision', 'Recall', 'Fscore', 'Consusion Matrix'])
            


def add_experiment_result(csv_content_list, name_csvfile):

    if os.path.exists(os.path.join(path, 'csvfiles', 'results', name_csvfile + '.csv')):
        with open(os.path.join(path, 'csvfiles', 'results', name_csvfile + '.csv'), 'a', newline='') as csvf:
            writer = csv.writer(csvf, delimiter=',')
            for i in range(len(csv_content_list)):
                writer.writerow(csv_content_list[i])
                

if(__name__=='__main__'):
    
    path = os.getcwd()
    name_csvfile = 'all_experiments'
    
    create_csv(path, name_csvfile)

    experiments_csv_list = (list(s for s in sorted(os.listdir(os.path.join(path, 'csvfiles'))) if s.endswith('.csv')))
    
    print(experiments_csv_list)
    
    for file in experiments_csv_list:
        print(file)
        csv_content_list = read_result(path, file)
        add_experiment_result(csv_content_list, name_csvfile)
    