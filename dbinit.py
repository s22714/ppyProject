import ssl

import pandas as pd
import sqlite3


def initialize_db():
    # ============================================IRIS====================================================
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width",
               "class"]
    df = pd.read_csv("data/iris.csv", names=headers, sep=';')

    conn = sqlite3.connect('clas.sqlite')
    query = f'Create table if not Exists IRIS (sepal_length Real, sepal_width Real, ' \
            f'petal_length real, petal_width real, class text)'
    conn.execute(query)
    df.to_sql("IRIS", conn, if_exists='replace', index=False)

    # ============================================WINE====================================================
    headers = ["class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash"
    , "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins"
    , "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"]

    df = pd.read_csv("data/wine.csv", names=headers)

    query = f'Create table if not Exists WINE (class real ,alcohol real, ' \
            f'malic_acid real, ash real, alcalinity_of_ash real, magnesium real, ' \
            f'total_phenols real, flavanoids real, nonflavanoid_phenols real, ' \
            f'proanthocyanins real, color_intensity real, hue real, ' \
            f'OD280_OD315_of_diluted_wines real, proline real)'
    conn.execute(query)
    df.to_sql("WINE", conn, if_exists='replace', index=False)

    # ============================================GLASS===================================================
    with open('data/glass.csv') as file:
        ncols = len(file.readline().split(','))

    headers = ["refractive_index", "Sodium", "Magnesium", "Aluminum"
        , "Silicon", "Potassium", "Calcium", "Barium", "Iron"
        , "class"]

    df = pd.read_csv("data/glass.csv", names=headers, usecols=range(1,ncols))

    query = f'Create table if not Exists GLASS (refractive_index real, ' \
            f'Sodium real, Magnesium real, Aluminum real, Silicon real, ' \
            f'Potassium real, Calcium real, Barium real, ' \
            f'Iron real, class real)'
    conn.execute(query)
    df.to_sql("GLASS", conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()


def get_data_from_db(picked):
    conn = sqlite3.connect('clas.sqlite')

    df = pd.read_sql(f"select * from {picked}", conn)
    conn.close()

    return df


def get_for_chart(col1,col2,picked):
    conn = sqlite3.connect('clas.sqlite')

    df = pd.read_sql(f"select class,{col1},{col2} from {picked}", conn)
    conn.close()

    return df


def get_columns_names(picked):
    if picked != "":
        conn = sqlite3.connect('clas.sqlite')
        cursor = conn.execute(f'select * from {picked}')
        cursor.close()
        conn.close()
        return list(map(lambda x: x[0], cursor.description))


def fetch_data(picked):
    if picked != "":
        conn = sqlite3.connect('clas.sqlite')
        cursor = conn.execute(f'select * from {picked}')
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result


def get_tables_names():
    conn = sqlite3.connect('clas.sqlite')
    sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    res = cursor.fetchall()
    cursor.close()
    conn.close()
    return res


def count_rows(picked):
    conn = sqlite3.connect('clas.sqlite')
    sql_query = f"SELECT COUNT(ALL) from {picked}"
    cursor = conn.cursor()
    cursor.execute(sql_query)
    res = cursor.fetchone()
    cursor.close()
    conn.close()
    return res


def save_to_tabel(picked,vec,cl):
    v = ()
    v2 = vec.split(',')
    conn = sqlite3.connect('clas.sqlite')
    cursor = conn.execute(f'select * from {picked}')
    cols = tuple(map(lambda x: x[0], cursor.description))
    num = 0
    for c in cols:
        if c == "class":
            v += (cl,)
        else:
            v += (v2[num],)
            num += 1

    query = f'''INSERT INTO {picked}{cols} VALUES {v}'''
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()


def num_of_col_in_file(filename,divider):
    with open(filename) as file:
        first_line = file.readline().strip('\n')
    line = first_line.split(divider)
    return len(line)


def import_from_file(headers,tablename,filename,separator,cls):
    ssl._create_default_https_context = ssl._create_unverified_context
    if cls == "First":
        headers[0] = "class"
    else:
        headers[-1] = "class"
    df = pd.read_csv(filename, names=headers, sep=separator)

    conn = sqlite3.connect('clas.sqlite')
    conn.execute(f'''create table {tablename} (temp text)''')
    conn.commit()
    for h in headers:
        if h == "class":
            conn.execute(f"alter table {tablename} add column {str(h)} text")
        else:
            conn.execute(f"alter table {tablename} add column {str(h)} real")
        conn.commit()
    df.to_sql(tablename, conn, if_exists='replace', index=False)
    conn.close()