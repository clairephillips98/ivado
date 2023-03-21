import sys
import process
import regression

if 'museum' in sys.argv:
    print('running museum processing')
    process.ProcessMuseumData()
if 'un_data' in sys.argv:
    print('running un data processing')
    process.ProcessUNData()
if 'regression' in sys.argv:
    print('running regression')
    regression.PreformLinearRegression()
