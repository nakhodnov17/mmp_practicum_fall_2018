with open('time_fit.csv', 'r') as fit_file, \
        open('time_predict.csv', 'r') as predict_file, \
        open('time_fit_predict.csv', 'r') as fit_predict_file, \
        open('table_time.csv', 'w') as result_file:
    header, _, _ = fit_file.readline(), predict_file.readline(), fit_predict_file.readline()
    result_file.write(header)
    for fit_line in fit_file:
        fit = fit_line.strip().split(',')
        predict = predict_file.readline().strip().split(',')
        fit_predict = fit_predict_file.readline().strip().split(',')
        result_file.write(
            ','.join(
                (
                    str(round(float(fit[_]), 2)) + '/' +
                    str(round(float(predict[_]), 2)) + '/' +
                    str(round(float(fit_predict[_]), 2)) if _ != 0
                    else str(fit[_]) for _ in range(len(fit))
                )
            ) + '\n'
        )
