import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from scipy import interpolate


def calculate_oiz_for_all_wells(
    df_history: pd.DataFrame,
    min_oiz,
    r_max,
    year_min,
    year_max
):
    wells_set = set(df_history['№ скважины'])
    df_reserves_based_on_history = pd.DataFrame()
    wells_with_error = []

    for well in wells_set:
        print(well)

        # считаем запасы на основе данных МЭР
        df_well = df_history.loc[df_history['№ скважины'] == well].reset_index(drop=True)
        # сначала пытаемся по всем точкам истории
        df_well_reserves = calculate_oiz_for_well_based_on_history(
            df_well=df_well,
            name_well=well,
            based_on='все точки истории'
        )[0]
        # если не получился результат по всем точкам, пытаемся по трём последним
        if df_well_reserves.empty:
            df_well_reserves = calculate_oiz_for_well_based_on_history(
                df_well=df_well,
                name_well=well,
                based_on='последние 3 точки истории'
            )[0]
            # если снова нет результата по запасам, запоминаем название скважины
            # (для неё будем считать НИЗ и ОИЗ интерполяцией по карте)
            if df_well_reserves.empty:
                wells_with_error.append(well)
                continue
        
        # перерасчёт ОИЗ успешно обработанных скважин с учётом ограничений (min_oiz, year_min, year_max)
        df_well_reserves = recalculate_oiz_using_restrictions(
            df_well_reserves=df_well_reserves,
            min_oiz=min_oiz,
            year_min=year_min,
            year_max=year_max
        )

        df_reserves_based_on_history = pd.concat([df_reserves_based_on_history, df_well_reserves])

    # расчёт НИЗ интерполяцией по карте
    # (для скважин, у которых не удалось получить результат на основе МЭР)
    print(wells_with_error)
    # координаты забоев всех скважин
    df_coordinates = df_history[[
        '№ скважины',
        'Координата забоя Х (по траектории)',
        'Координата забоя Y (по траектории)'
    ]]
    df_coordinates = df_coordinates.drop_duplicates(subset=['№ скважины']).reset_index(drop=True)
    df_coordinates.set_index('№ скважины', inplace=True)

    # координаты забоев и значения НИЗ для скважин с найденными (на основе МЭР) НИЗ
    df_already_calculated = df_reserves_based_on_history.set_index('Скважина')
    df_field = pd.merge(
        df_coordinates[[
            'Координата забоя Х (по траектории)',
            'Координата забоя Y (по траектории)'
        ]],
        df_already_calculated[['НИЗ']], left_index=True, right_index=True
    )

    # координаты забоев скважин, для которых не удалось рассчитать НИЗ на основе МЭР
    df_with_errors = pd.DataFrame({'Скважина': wells_with_error, '№ скважины': wells_with_error})
    df_with_errors.set_index('№ скважины', inplace=True)
    df_with_errors = pd.merge(
        df_coordinates[[
            'Координата забоя Х (по траектории)',
            'Координата забоя Y (по траектории)'
        ]],
        df_with_errors[['Скважина']], left_index=True, right_index=True
    )

    new_niz = []
    new_oiz =[]
    warns = []

    for well in wells_with_error:

        df_well = df_history.loc[df_history['№ скважины'] == well].reset_index(drop=True)
        df_well['Накопленная добыча нефти, т'] = df_well['Добыча нефти за посл.месяц, т'].cumsum()

        new_niz_int, new_oiz_int, warn = calculate_oiz_for_well_based_on_map(
            name_well=well,
            df_well=df_well,
            df_with_errors=df_with_errors,
            df_field=df_field,
            r_max=r_max,
            min_oiz=min_oiz,
            year_min=year_min,
            year_max=year_max
        )
        new_niz.append(new_niz_int)
        new_oiz.append(new_oiz_int)
        warns.append(warn)
    
    df_with_errors['НИЗ'] = new_niz
    df_with_errors['ОИЗ'] = new_oiz
    df_with_errors['Предупреждение'] = warns

    df_all_reserves = pd.concat([
        df_with_errors[['Скважина', 'ОИЗ']],
        df_reserves_based_on_history[['Скважина', 'ОИЗ']]
    ])
    df_all_reserves['ОИЗ'] = df_all_reserves['ОИЗ'] / 1000

    df_reserves_based_on_history = df_reserves_based_on_history.set_index('Скважина')
    df_with_errors = df_with_errors.set_index('Скважина')

    write_reserves_to_excel(
        df_reserves_based_on_history=df_reserves_based_on_history,
        df_with_errors=df_with_errors
    )

    return df_all_reserves


def calculate_oiz_for_well_based_on_history(
    df_well: pd.DataFrame,
    name_well: str,
    based_on='все точки истории'
) -> tuple:

    df_well = prepare_df_for_statistical_methods(df_well)
    error = ''

    q_before_the_last = 0

    match based_on:
        case 'все точки истории':
            if len(df_well['Накопленная добыча нефти, т']) > 1:
                q_before_the_last = float(df_well['Добыча нефти за посл.месяц, т'][-2:-1])
            else:
                error = 'имеется только одна точка'
        case 'последние 3 точки истории':
            if len(df_well['Накопленная добыча нефти, т']) > 2:
                df_well = df_well.tail(3)
                q_last = float(df_well['Добыча нефти за посл.месяц, т'][-1:])
                q_before_the_last = float(df_well['Добыча нефти за посл.месяц, т'][-2:-1])
                if q_last / q_before_the_last < 0.25:
                    df_well = df_well[:-1]
            else:
                error = 'имеется только одна или две точки'

    # построение моделей на основе статистических методов
    models = []  # list of tuples; (niz, oiz, correlation, determination)
    methods = ['Nazarov_Sipachev', 'Sipachev_Pasevich', 'FNI', 'Maksimov', 'Sazonov']
    for name in methods:
        models.append(linear_model_with_given_statistical_method(
            df_well=df_well,
            method=name
        ))

    # формирование итогового датафрейма
    df_well_result = create_df_with_reserves(
        name_well=name_well,
        df_well=df_well,
        methods=methods,
        models=models,
        q_before_the_last=q_before_the_last
    )

    # проверка на возможные ошибки в итоговом датафрейме
    df_well_result, check = check_calculated_reserves(df_well_result)
    if check:
        error = check
    
    df_well_result = df_well_result.sort_values('ОИЗ')
    df_well_result = df_well_result.tail(1)

    # запись информации о количестве использованных точек МЭР при расчёте
    if not df_well_result.empty:
        if based_on == 'все точки истории':
            df_well_result['Предупреждение'] = 'Расчёт по всем точкам истории'
        else:
            df_well_result['Предупреждение'] = 'Расчёт по последним 3-м точкам истории'
    
    return df_well_result, error


def recalculate_oiz_using_restrictions(
    df_well_reserves: pd.DataFrame,
    min_oiz,
    year_min,
    year_max
):
    new_oiz = df_well_reserves['ОИЗ']
    if df_well_reserves['Оставшееся время работы, прогноз, лет'].values[0] > year_max:
        new_oiz = (df_well_reserves['Добыча нефти за посл. мес работы скв., т'] +
                    df_well_reserves['Добыча нефти за предпосл. мес работы скв., т']) * year_max * 6
    elif df_well_reserves['Оставшееся время работы, прогноз, лет'].values[0] < year_min:
        new_oiz = (df_well_reserves['Добыча нефти за посл. мес работы скв., т'] +
                    df_well_reserves['Добыча нефти за предпосл. мес работы скв., т']) * year_min * 6
    if df_well_reserves['ОИЗ'].values[0] < min_oiz:
        new_oiz = min_oiz
        
    df_well_reserves['ОИЗ'] = new_oiz
    df_well_reserves['Оставшееся время работы, прогноз, лет'] = new_oiz / \
        (df_well_reserves['Добыча нефти за посл. мес работы скв., т'] * 12)
    
    return df_well_reserves


def prepare_df_for_statistical_methods(
    df_well: pd.DataFrame
) -> pd.DataFrame:

    # TODO: проверить, замедлит ли работу df_prepared = df_well.copy()

    df_well['Накопленная добыча нефти, т'] = df_well['Добыча нефти за посл.месяц, т'].cumsum()
    df_well['Накопленная добыча жидкости, т'] = df_well['Добыча жидкости за посл.месяц, т'].cumsum()
    df_well['Накопленная добыча воды, т'] = df_well['Накопленная добыча жидкости, т'] - \
        df_well['Накопленная добыча нефти, т']

    df_well['Отношение накопленной добычи жидкости к накопленной добыче нефти'] = \
        df_well['Накопленная добыча жидкости, т'] / df_well['Накопленная добыча нефти, т']

    df_well['Логарифм накопленной добычи жидкости'] = np.log(df_well['Накопленная добыча жидкости, т'])
    df_well['Логарифм накопленной добычи воды'] = np.log(df_well['Накопленная добыча воды, т'])
    df_well['Логарифм накопленной добычи нефти'] = np.log(df_well['Накопленная добыча нефти, т'])

    df_well['Год'] = df_well['Дата'].map(lambda x: x.year)

    return df_well


def linear_model_with_given_statistical_method(
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
            y = df_well['Логарифм накопленной добычи воды']
        case 'Sazonov':
            x = df_well['Накопленная добыча нефти, т'].values.reshape((-1, 1))
            y = df_well['Логарифм накопленной добычи жидкости']
    
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


def check_calculated_reserves(
    df_well_result: pd.DataFrame
):
    error = None

    df_well_result = df_well_result.loc[df_well_result['ОИЗ'] > 0]
    if df_well_result.empty:
        error = 'Остаточные запасы <= 0'
    
    df_up = df_well_result.loc[df_well_result['Correlation'] > 0.7]
    df_down = df_well_result.loc[df_well_result['Correlation'] < (-0.7)]
    df_well_result = pd.concat([df_up, df_down]).reset_index()
    if df_well_result.empty:
        error = 'Корреляция <0.7 или >-0.7'
    
    df_well_result = df_well_result.loc[df_well_result['Оставшееся время работы, прогноз, лет'] < 50]
    if df_well_result.empty:
        error = 'Оставшееся время работы превышает 50 лет'
    
    return df_well_result, error


def create_df_with_reserves(
    name_well: str,
    df_well: pd.DataFrame,
    methods: list,
    models: list,
    q_before_the_last
):
    df_well_result = pd.DataFrame()
    df_well_result['НИЗ'] = [model[0] for model in models]
    df_well_result['ОИЗ'] = [model[1] for model in models]
    df_well_result['Метод'] = methods
    df_well_result['Добыча нефти за посл. мес работы скв., т'] = df_well['Добыча нефти за посл.месяц, т'].values[-1]
    df_well_result['Добыча нефти за предпосл. мес работы скв., т'] = q_before_the_last
    df_well_result['Накопленная добыча нефти, т'] = df_well['Накопленная добыча нефти, т'].values[-1]
    df_well_result['Скважина'] = name_well
    df_well_result['Correlation'] = [model[2] for model in models]
    df_well_result['Sigma'] = [model[3] for model in models]
    df_well_result['Оставшееся время работы, прогноз, лет'] = \
        df_well_result['ОИЗ'] / (df_well_result['Добыча нефти за посл. мес работы скв., т'] * 12)
    df_well_result['Время работы, прошло, лет'] = int(df_well['Год'].tail(1)) - int(df_well['Год'].head(1))
    df_well_result['Координата X'] = float(df_well['Координата забоя Х (по траектории)'][-1:])
    df_well_result['Координата Y'] = float(df_well['Координата забоя Y (по траектории)'][-1:])

    return df_well_result


def calculate_oiz_for_well_based_on_map(
    name_well,
    df_well,
    df_with_errors,
    df_field,
    r_max,
    min_oiz,
    year_min,
    year_max,
):
    warns = []
    
    x = df_with_errors['Координата забоя Х (по траектории)'][name_well]
    y = df_with_errors['Координата забоя Y (по траектории)'][name_well]

    distance = ((x - df_field['Координата забоя Х (по траектории)']) ** 2 +
                (y - df_field['Координата забоя Y (по траектории)']) ** 2) ** 0.5
    r_min = distance.min()
    if r_min > r_max:
        warns.append('! Ближайшая скважина на расстоянии ' + str(r_min))
    else:
        warns.append('Скважина в пределах ограничений')
    
    gur = interpolate_gur(
            x=x,
            y=y,
            table_x=df_field[['Координата забоя Х (по траектории)']],
            table_y=df_field[['Координата забоя Y (по траектории)']],
            table_z=df_field[['НИЗ']]
        )
    
    if len(df_well['Накопленная добыча нефти, т']) > 1:
        # добыча нефти за предпоследний месяц истории
        q_before_the_last = float(df_well['Добыча нефти за посл.месяц, т'][-2:-1])
    else:
        q_before_the_last = 0
        
    # добыча нефти за последний месяц истории
    q_last = df_well['Добыча нефти за посл.месяц, т'].values[-1]

    # накопленная добыча нефти за всю историю работы скважины
    cumulative_oil_production = df_well['Накопленная добыча нефти, т'].values[-1]

    new_oiz_list = []

    for k in gur:
        oiz = k - cumulative_oil_production
        if oiz > 0:
            new_oiz_list.append(oiz)

    if len(new_oiz_list) == 0:
        new_oiz = (q_before_the_last + q_last) * year_min * 6
    elif len(new_oiz_list) == 1:
        new_oiz = new_oiz_list[0]
        forecast_residual_operation_time = new_oiz / (q_last * 12)
        if forecast_residual_operation_time > year_max:
            new_oiz = (q_before_the_last + q_last) * year_max * 6
        elif forecast_residual_operation_time  < year_min:
            new_oiz = (q_before_the_last + q_last) * year_min * 6
    else:
        forecast_residual_operation_time_1 = new_oiz_list[0] / (q_last * 12)
        forecast_residual_operation_time_2 = new_oiz_list[1] / (q_last * 12)
        if forecast_residual_operation_time_1 < year_max and forecast_residual_operation_time_1 > year_min:
            new_oiz = new_oiz_list[0]
        else:
            if forecast_residual_operation_time_2 < year_max and forecast_residual_operation_time_2 > year_min:
                new_oiz = new_oiz_list[1]
            else:
                if forecast_residual_operation_time_1 > year_max:
                    new_oiz = (q_before_the_last + q_last) * year_max * 6
                elif forecast_residual_operation_time_1 < year_min:
                    new_oiz = (q_before_the_last + q_last) * year_min * 6
    if new_oiz < min_oiz:
        new_oiz = min_oiz
    new_oiz_int = int(new_oiz)
    new_niz_int = int(new_oiz + cumulative_oil_production)

    return new_niz_int, new_oiz_int, warns


def interpolate_gur(
    x,
    y,
    table_x,
    table_y,
    table_z
) -> tuple:

    table_x = np.reshape(np.array(table_x, dtype='float64'), (-1,))
    table_y = np.reshape(np.array(table_y, dtype='float64'), (-1,))
    table_z = np.reshape(np.array(table_z, dtype='float64'), (-1,))

    if len(table_x) <= 16:
        gur_1 = interpolate.interp2d(table_x, table_y, table_z, kind='linear')
        gur_1 = gur_1(x, y)[0]
        gur_2 = gur_1
    else:
        gur_1 = interpolate.griddata(
            (table_x, table_y),
            table_z,
            (x, y),
            method='cubic'
        )
        gur_2 = interpolate.interp2d(table_x, table_y, table_z, kind='cubic')
        gur_2 = gur_2(x, y)[0]
    
    return gur_1, gur_2


def write_reserves_to_excel(
    df_reserves_based_on_history: pd.DataFrame,
    df_with_errors: pd.DataFrame
):
    with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), 'data', 'Подсчёт ОИЗ.xlsx')) as writer:
        df_reserves_based_on_history.to_excel(
            writer,
            sheet_name='Расчёт по истории',
            startrow=0,
            startcol=0,
            header=True,
            index=True
        )
        df_with_errors.to_excel(
            writer,
            sheet_name='Расчёт по карте',
            startrow=0,
            startcol=0,
            header=True,
            index=True
        )
