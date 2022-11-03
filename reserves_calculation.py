import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_reserves(

):
    # TODO: реализовать расчёт НИЗ и ОИЗ для каждой из скважин
    pass


def calculate_reserves_statistics(
    df_well: pd.DataFrame,
    name_well: str,
    based_on='все точки'
) -> tuple:

    df_well = prepare_df(df_well)
    error = ''

    q_before_the_last = 0

    match based_on:

        case 'все точки':
            if len(df_well['Накопленная добыча нефти, т']) > 1:
                q_before_the_last = float(df_well['Добыча нефти за посл.месяц, т'][-2:-1])
            else:
                error = 'имеется только одна точка'

        case 'последние 3 точки':
            if len(df_well['Накопленная добыча нефти, т']) > 2:
                df_well = df_well.tail(3)
                q_last = float(df_well['Добыча нефти за посл.месяц, т'][-1:])
                q_before_the_last = float(df_well['Добыча нефти за посл.месяц, т'][-2:-1])
                if q_last / q_before_the_last < 0.25:
                    df_well = df_well[:-1]
            else:
                error = 'имеется только одна или две точки'
    
    cumulative_oil_production = df_well['Накопленная добыча нефти, т'].values[-1]
    well_operation_time = int(df_well['Год'].tail(1)) - int(df_well['Год'].head(1))

    # статистические методы
    models = []  # list of tuples; (reserves, residual_reserves, korrelation, determination)
    methods = ['Nazarov_Sipachev', 'Sipachev_Pasevich', 'FNI', 'Maksimov', 'Sazonov']
    for name in methods:
        models.append(linear_model_with_given_method(
            df=df_well,
            method=name
        ))

    # формирование итогового датафрейма
    df_well_result = pd.DataFrame()
    df_well_result['НИЗ'] = [model[0] for model in models]
    df_well_result['ОИЗ'] = [model[1] for model in models]
    df_well_result['Метод'] = methods
    df_well_result['Добыча нефти за посл. мес работы скв., т'] = df_well['Добыча нефти за посл.месяц, т'].values[-1]
    df_well_result['Добыча нефти за предпосл. мес работы скв., т'] = q_before_the_last
    df_well_result['Накопленная добыча нефти, т'] = cumulative_oil_production
    df_well_result['Скважина'] = name_well
    df_well_result['Correlation'] = [model[2] for model in models]
    df_well_result['Sigma'] = [model[3] for model in models]
    df_well_result['Оставшееся время работы, прогноз, лет'] = \
        df_well_result['ОИЗ'] / (df_well_result['Добыча нефти за посл. мес работы скв., т'] * 12)
    df_well_result['Время работы, прошло, лет'] = well_operation_time
    df_well_result['Координата X'] = float(df_well['Координата забоя Х (по траектории)'][-1:])
    df_well_result['Координата Y'] = float(df_well['Координата забоя Y (по траектории)'][-1:])

    # проверка на ошибки
    check = check_reserves_statistics(df_well_result)
    if check:
        error = check
    
    df_well_result = df_well_result.sort_values('ОИЗ')
    df_well_result = df_well_result.tail(1)

    if not df_well_result.empty:
        if based_on == 'все точки':
            df_well_result['Метка'] = 'Расчёт по всем точкам'
        else:
            df_well_result['Метка'] = 'Расчёт по последним 3-м точкам'
    
    return df_well_result, error


def prepare_df(
    df_well: pd.DataFrame
) -> pd.DataFrame:

    # TODO: проверить, замедлит ли работу df_prepared = df_well.copy()

    df_well['Накопленная добыча нефти, т'] = df_well['Добыча нефти за посл.месяц, т'].cumsum()
    df_well['Накопленная добыча жидкости, т'] = df_well['Добыча жидкости за посл.месяц, т'].cumsum()
    df_well['Накопленная добыча воды, т'] = df_well['Накопленная добыча жидкости, т'] - \
        df_well['Накопленная добыча нефти, т']

    df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти'] = \
        df_well['Накопленная добыча жидкости, т'] / df_well['Накопленная добыча нефти, т']

    df_well['Логарифм накопленной добычи жидкости, т'] = np.log(df_well['Накопленная добыча жидкости, т'])
    df_well['Логарифм накопленной добычи воды, т'] = np.log(df_well['Накопленная добыча воды, т'])
    df_well['Логарифм накопленной добычи нефти, т'] = np.log(df_well['Накопленная добыча нефти, т'])

    df_well['Год'] = df_well['Дата'].map(lambda x: x.year)

    return df_well


def linear_model_with_given_method(
    df_well: pd.DataFrame,
    method: str
) -> tuple:
    match method:
        case 'Nazarov_Sipachev':
            x = df_well['Накопленная добыча воды, т'].values.reshape((-1, 1))
            y = df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти']
        case 'Sipachev_Pasevich':
            x = df_well['Накопленная добыча жидкости, т'].values.reshape((-1, 1))
            y = df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти']
        case 'FNI':
            x = df_well['Накопленная добыча нефти, т'].values.reshape((-1, 1))
            y = df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти']
        case 'Maksimov':
            x = df_well['Накопленная добыча нефти, т'].values.reshape((-1, 1))
            y = df_well['Логарифм накопленной добычи воды, т']
        case 'Sazonov':
            x = df_well['Накопленная добыча нефти, т'].values.reshape((-1, 1))
            y = df_well['Логарифм накопленной добычи жидкости, т']
    
    model = LinearRegression().fit(x, y)
    a = model.coef_
    b = model.intercept_
    a = float(a)
    a = np.fabs(a)
    cumulative_oil_production = df_well['Накопленная добыча нефти, т'].values[-1]
    if a != 0:
        match method:
            case 'Nazarov_Sipachev':
                niz = (1 / a) * (1 - ((b - 1) * (1 - 0.99) / 0.99) ** 0.5)
            case 'Sipachev_Pasevich':
                niz = (1 / a) - ((0.01 * b) / (a ** 2)) ** 0.5
            case 'FNI':
                niz = 1 / (2 * a * (1 - 0.99)) - b / 2 * a
            case 'Maksimov' | 'Sazonov':
                niz = (1 / a) * np.log(0.99 / ((1 - 0.99) * a * np.exp(b)))
        oiz = niz - cumulative_oil_production  # остаточные извлекаемые запасы нефти
    else:
        niz = 0
        oiz = 0
    correlation = np.fabs(np.corrcoef(
        df_well['Накопленная добыча воды, т'],
        df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти']
    )[1, 0])
    determination = model.score(x, y)

    return niz, oiz, correlation, determination


def check_reserves_statistics(
    df_well_result: pd.DataFrame
):
    error = None

    df_well_result = df_well_result.loc[df_well_result['ОИЗ'] > 0]
    if df_well_result.empty:
        error = 'Остаточные запасы <= 0'
    
    df_up = df_well_result.loc[df_well_result['Korrelation'] > 0.7]
    df_down = df_well_result.loc[df_well_result['Korrelation'] < (-0.7)]
    df_well_result = pd.concat([df_up, df_down]).reset_index()
    if df_well_result.empty:
        error = 'Корреляция <0.7 или >-0.7'
    
    df_well_result = df_well_result.loc[df_well_result['Оставшееся время работы, прогноз, лет'] < 50]
    if df_well_result.empty:
        error = 'Оставшееся время работы превышает 50 лет'
    
    return error
