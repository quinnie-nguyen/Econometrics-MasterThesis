import utils
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

month = ['Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec']

year = [17, 18, 19, 20, 21, 22, 23, 24]

t2m = [6/12, 3/12, 1/12]
def main():
    for mm in month:
        for yy in year:
            contract = f"{mm}-{yy}"
            print(contract)
            for tt in t2m:
                print(tt)
                egarch_obj = utils.calibration_CNT(contract=contract, t2m=tt, model='EGARCH', delta_hegde=False, estimation_length=3)
                egarch_obj.archive_params()
                egarch_obj.archive_crps()


if __name__ == '__main__':
    main()