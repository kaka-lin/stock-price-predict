import getopt, sys, os
import csv
import pandas as pd
import locale
from locale import atof


locale.setlocale(locale.LC_NUMERIC, '')

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v:f:", ["help", "output=", "filepath"])
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)
    output = None
    verbose = False
    filepath = os.getcwd()

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-f", "--filepath"):
            filepath = a
        else:
            assert False, "unhandled option"

    return filepath

def usage():
    print ("=======================\n"\
           "please input filepath\n"\
           "ex: python get_feature.py -f ./data/20180427 \n"\
           "=======================")


def get_feature_data(filepath, encode=None, **argv):
    """
        input:
          filepath
          encode
          argv:
            Code,Date,CB,Open,High,Low,Close,Volumn
            True or False
    """
    params = []
    for param in argv:
        params = [i for i, t in argv.items() if t == True]

    # abs filepath
    filepath = os.path.abspath(filepath)
    get_date = os.path.basename(filepath)

    tetfp_file = os.path.join(filepath, "tetfp.csv")
    save_process_path = os.path.join(os.path.abspath("./data/" + get_date + "_process"))

    with open(tetfp_file, encoding=encode) as file:
        rows = csv.reader(file, delimiter=",")
        data = []
        for row in rows:
            new_index = []
            for index in row:
                if index:
                    index = index.strip()
                    new_index.append(index)
            data.append(new_index)

    df = pd.DataFrame(data=data[1:], columns=change_columns(*data[0]))
    df = df.dropna()
    df["Volumn"] = pd.to_numeric(df["Volumn"].replace('\.','', regex=True)
                                             .replace(',','', regex=True)
                                             .astype(int))
    types = set(df.loc[:,"Code"])

    if not os.path.exists(save_process_path):
        os.mkdir(save_process_path)
    for t in types:
        str_t = str(int(t))
        t_types = df.loc[df['Code'] == t][params]
        t_types.to_csv(os.path.join(save_process_path, get_date + "_" + str_t + ".csv"), index=False)

def change_columns(*header):
    """
        replace header to English
    """
    column_dict = {
        "代碼":"Code",
        "日期":"Date",
        "中文簡稱":"CB",
        "開盤價(元)":"Open",
        "最高價(元)":"High",
        "最低價(元)":"Low",
        "收盤價(元)":"Close",
        "成交張數(張)": "Volumn"
    }
    return [column_dict[h] for h in header]

if __name__ == "__main__":

    """
        choose data output column
    """
    choose = {
        "Code":True,
        "Date":True,
        "CB": False,
        "Open": True,
        "High": True,
        "Low": True,
        "Close": True,
        "Volumn": True
    }
    filepath = main()
    get_feature_data(filepath, "big5", **choose)
