import datetime

data_dir = "E:/Users/KNOL/Documents/python33/invester/predict_usin_mini1min/data/"

input_list = [
    "N225minif_2016.xlsx - 1min.csv",
    "N225minif_2017.xlsx - 1min.csv",
    "N225minif_2018.xlsx - 1min.csv",
    "N225minif_2019.xlsx - 1min.csv",
    "N225minif_2020.xlsx - 1min.csv",
]

renamed_columns = {'時間':'Time', '始値':'Open',  '高値':'High', '安値':'Low' , '終値':'Close', '出来高':'Volume'}
renamed_index_name = 'Date'

timesteps = 50

input_columns = [
    'diff_Average',
    'Volume'
]

label_columns = [
    'future_rsi',
]

def conv(temp):
    H, M = map(int, temp.split(':'))
    return datetime.timedelta(hours=H, minutes=M)

bar_time = '5T'