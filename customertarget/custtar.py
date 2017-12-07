import pandas as pd
from sklearn import ensemble
from sklearn.feature_extraction import DictVectorizer
import locale
import datetime
import numpy as np

# xl = pd.ExcelFile('C:\\Users\\Chetan.Chougle\\Desktop\\Book1.xlsx')
# df = xl.parse("Sheet1")
# print(df.head())
def predict_heuristic(previous_predict, month_predict, actual_previous_value, actual_value):
    # print("here3")
    # print("here6")
    # print(np.any( actual_value < month_predict))
    # print("her5")
    # print("actual")
    # print(actual_value)
    # print("month_predict")
    # print(month_predict)
    # print(actual_value < month_predict)
    if (
          actual_value < month_predict and
         abs((actual_value - month_predict) / month_predict) > .3):
        if (
             actual_previous_value > previous_predict and
             abs((previous_predict - actual_previous_value) / actual_previous_value) > .3):
            return False
        else:
            return True
    else:
        return False

def get_dataframe():
    xl = pd.ExcelFile('C:\\Users\\Chetan.Chougle\\Desktop\\Book2.xlsx')
    # df = xl.parse("Sheet1")
    # rows = df

    df = pd.DataFrame.from_records(
        xl.parse("Sheet1"),
        columns=['CustomerName', 'Sales', 'Month', 'Year','PreviousMonthSales']
    )
    df["CustomerName"] = df["CustomerName"].astype('category')
    # print(df)
    # print(df['CustomerName'].tolist())
    return df

def missed_customers():
    """ Returns a list of tuples of the customer name, the prediction, and
        the actual amount that the customer has bought.
    """

    raw = get_dataframe()
    vec = DictVectorizer()
    today = datetime.date.today()
    currentMonth = today.month
    currentYear = today.year
    lastMonth = (today.replace(day=1) - datetime.timedelta(days=1)).month
    lastMonthYear = (today.replace(day=1) - datetime.timedelta(days=1)).year
    results = []

    # Exclude this month's value
    df = raw.loc[(raw['Month'] != currentMonth) & (raw['Year'] != currentYear)]

    for customer in set(df['CustomerName'].tolist()):
        # compare this month's real value to the prediction
        actual_value = 0.0
        actual_previous_value = 0.0
        # print("here2")
        # Get the actual_value and actual_previous_value
        # print("sddjd")
        # print(raw.loc[(raw['CustomerName'] == customer) & (raw['Year'] ==currentYear ) ]['Sales'])
        # print("sdfs")
        # new_raw = raw.loc[(raw['CustomerName'] == customer)     , 'Sales']
        # new_raw2 = new_raw.loc[(raw['Year'] == currentYear)   ]
        # print( new_raw.iloc[0] )
        # print( raw.loc[(raw['CustomerName'] == customer )['Sales']])
        # print("\n")
        # print("Current year")
        # print(currentYear)
        # print("currentMonth")
        # print(currentMonth)
        # print("last month")
        # print(lastMonth)
        # print("lastMonthYear")
        # print(lastMonthYear)
        print("sales")
        print(raw.loc[(raw['CustomerName'] == customer)     , 'Sales'].iloc[0] )
        actual_previous_value = raw.loc[(raw['CustomerName'] == customer)     , 'PreviousMonthSales'].iloc[0]
        actual_value = raw.loc[(raw['CustomerName'] == customer)     , 'Sales'].iloc[0]

        # Transforming Data
        temp = df.loc[df['CustomerName'] == customer]
        targets = temp['Sales']
        del temp['CustomerName']
        del temp['Sales']
        records = temp.to_dict(orient="records")
        vec_data = vec.fit_transform(records).toarray()
        # print("previous1")
        # Fitting the regressor, and use all available cores
        regressor = ensemble.RandomForestRegressor(n_jobs=-1)
        regressor.fit(vec_data, targets)
        # print("previous2")
        # Predict the past two months using the regressor
        previous_predict = regressor.predict(vec.transform({
            'Year': lastMonthYear,
            'Month': lastMonth
        }).toarray())[0]
        # print("previous3")
        month_predict = regressor.predict(vec.transform({
            'Year': currentYear,
            'Month': currentMonth
        }).toarray())[0]
        print("actual value")
        print(actual_value)
        print("month_predict")
        print(month_predict)
        # print("previous")
        # print(previous_predict)
        # predict_heuristic(previous_predict, month_predict, actual_previous_value, actual_value)
        if(predict_heuristic(previous_predict, month_predict, actual_previous_value, actual_value)):
            print("hisd")
            results.append((
                customer,
                month_predict,
                actual_value
            ))

    return results

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    customers = missed_customers()
    print("here")
    # print(customers)
    for customer in  set(customers):
        print("{} was predicted to buy around {}, they bought only {}".format(
            customer[0],
            locale.currency(customer[1], grouping=True),
            locale.currency(customer[2], grouping=True)
        ))
