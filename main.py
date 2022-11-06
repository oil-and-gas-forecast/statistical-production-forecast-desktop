import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, NonlinearConstraint, fsolve
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QLabel, QFileDialog
from wells_history_preprocessing import history_preprocessing
from reserves_calculation import calculate_oiz_for_all_wells
from displacement_characteristic_model import DisplacementCharacteristic


def choose_file_with_monthly_operating_report():
    file_path, check = QFileDialog.getOpenFileName(
        parent=None,
        caption='Выберите файл с МЭР',
        directory=os.path.dirname(__file__),
        filter='Excel files (*.xlsx)'
    )
    if check:
        df_initial = pd.read_excel(file_path, sheet_name='МЭР')
        df_boundary = pd.read_excel(file_path, sheet_name='ОИЗ')
        main_calculations(df_initial, df_boundary)


def main_calculations(
    df_initial: pd.DataFrame,
    df_boundary: pd.DataFrame
):
    df_initial = history_preprocessing(df_initial, max_delta=365)

    oiz_dict, boundaries_dict, wells_data_dict = create_dicts_with_essential_data(
        df_initial=df_initial,
        df_boundary=df_boundary
    )

    q_cutoff = df_boundary['Дебит отсечки'].values[0]

    output = {}
    for well in wells_data_dict.keys():
        output[well] = []

    for well in wells_data_dict.keys():
        oil_rates = np.array(wells_data_dict[well][0], dtype='float')
        fluid_rates = np.array(wells_data_dict[well][1], dtype='float')
        time_in_prod = np.array(wells_data_dict[well][2], dtype='float')

        dates = []
        for g in range(oil_rates.size):
            y = int(str(wells_data_dict[well][3][g])[0:4])
            m = int(str(wells_data_dict[well][3][g])[5:7])
            dates.append([y, m, 1])

        try:
            oiz_for_well = oiz_dict[well][0]
        except:
            output[well] = [None] * 9
            continue

        corey_oil_left = boundaries_dict[well][2]
        if np.isnan(corey_oil_left):
            corey_oil_left = -np.inf
        corey_oil_right = np.inf

        corey_water_left = boundaries_dict[well][3]
        if np.isnan(corey_water_left):
            corey_water_left = -np.inf
        corey_water_right = np.inf

        mef_left = boundaries_dict[well][4]
        if np.isnan(mef_left):
            mef_left = -np.inf
        mef_right = boundaries_dict[well][5]
        if np.isnan(mef_right):
            mef_right = np.inf

        q_oil_sum_for_well = np.sum(oil_rates) / 1000
        niz_for_well = oiz_for_well + q_oil_sum_for_well

        watercut_fact = (fluid_rates - oil_rates) / fluid_rates
        watercut_fact[watercut_fact == -np.inf] = 0
        watercut_fact[watercut_fact == np.inf] = 0
        current_recovery_factor = q_oil_sum_for_well / niz_for_well

        check_watercuts = len(watercut_fact[np.where(watercut_fact < watercut_fact[-1])]) > \
                          len(watercut_fact[np.where(watercut_fact > watercut_fact[-1])])
        if check_watercuts:
            new_watercut_fact = watercut_fact
            new_oil_rates = oil_rates
            new_fluid_rates = fluid_rates
        else:
            new_watercut_fact = watercut_fact[np.where(watercut_fact <= watercut_fact[-1])]
            new_oil_rates = oil_rates[np.where(watercut_fact <= watercut_fact[-1])]
            new_fluid_rates = fluid_rates[np.where(watercut_fact <= watercut_fact[-1])]

        corey_oil = 3
        corey_water = 2
        mef = 3
        params_list = [corey_oil, corey_water, mef]
        displacement_characteristic_model = DisplacementCharacteristic(
            oil_production=new_oil_rates,
            liq_production=new_fluid_rates,
            niz=niz_for_well,
            considerations=boundaries_dict,
            well_name=well,
            mark=check_watercuts,
            wc_fact=new_watercut_fact,
            rf_now=current_recovery_factor
        )
        bnds = Bounds(
            [corey_oil_left, corey_water_left, mef_left],
            [corey_oil_right, corey_water_right, mef_right]
        )
        try:
            nonlinear_con = NonlinearConstraint(
                displacement_characteristic_model.to_conditions,
                [-0.00001],
                [0.00001]
            )
            res = minimize(
                displacement_characteristic_model.solver,
                params_list,
                method='trust-constr',
                bounds=bnds,
                constraints=nonlinear_con
            )
            params_list = res.x
            output[well] = res.x
        except:
            output[well] = ['Невозможно'] * 3

        if check_watercuts:
            marker_text = '-'
        else:
            marker_text = 'низкое качество данных'

        # проверка характеристик вытеснения
        if params_list[0] < 0.01 and params_list[1] < 0.01 and marker_text == '-':
            marker_text = 'отброшены точки выше последней по обводнённости'
            new_watercut_fact = watercut_fact[np.where(watercut_fact <= watercut_fact[-1])]
            new_oil_rates = oil_rates[np.where(watercut_fact <= watercut_fact[-1])]
            new_fluid_rates = fluid_rates[np.where(watercut_fact <= watercut_fact[-1])]
            corey_oil = 3
            corey_water = 2
            mef = 3
            params_list = [corey_oil, corey_water, mef]
            displacement_characteristic_model = DisplacementCharacteristic(
                oil_production=new_oil_rates,
                liq_production=new_fluid_rates,
                niz=niz_for_well,
                considerations=boundaries_dict,
                well_name=well,
                mark=check_watercuts,
                wc_fact=new_watercut_fact,
                rf_now=current_recovery_factor
            )
            try:
                nonlinear_con = NonlinearConstraint(
                    displacement_characteristic_model.to_conditions,
                    [-0.00001],
                    [0.0001]
                )
                res = minimize(
                    displacement_characteristic_model.solver,
                    params_list,
                    method='trust-constr',
                    bounds=bnds,
                    constraints=nonlinear_con
                )
                params_list = res.x
                output[well] = res.x
            except:
                output[well] = ['Невозможно'] * 3

        if (params_list[0] < 0.01 and params_list[1] < 0.01) or output[well][0] == 'Невозможно':
            marker_text = 'отброшены все точки кроме последней'

            output[well] = np.array([3, 2, 3])

            last_watercut = watercut_fact[-1]
            func = lambda x: last_watercut * x ** 3 + 3 * (2 * last_watercut - 1) * x ** 2 - \
                             3 * last_watercut * x + last_watercut
            recovery_factor_new = fsolve(func, (0))
            oiz_for_well = q_oil_sum_for_well / recovery_factor_new[0] - q_oil_sum_for_well
            oiz_dict[well] = [oiz_for_well]
            niz_for_well = oiz_for_well + q_oil_sum_for_well
            current_recovery_factor = q_oil_sum_for_well / niz_for_well

        output[well] = [output[well][0], output[well][1], output[well][2],
                        marker_text, [fluid_rates], [oil_rates],
                        [time_in_prod], [dates], current_recovery_factor]
    print(output)


def create_dicts_with_essential_data(
    df_initial,
    df_boundary
) -> tuple:
    oiz_dict = calculate_oiz_for_all_wells(df_initial, 2000, 1000, 5, 50).set_index('Скважина').T.to_dict('list')

    df_boundary = df_boundary.drop(df_boundary.columns[1], axis='columns')
    boundaries_dict = df_boundary.set_index('№ скв.').T.to_dict('list')

    time_in_production = df_initial['Время работы в добыче, часы'].ravel()
    wells = df_initial['№ скважины'].ravel()
    last_oil_rate = df_initial['Добыча нефти за посл.месяц, т'].ravel()
    last_fluid_rate = df_initial['Добыча жидкости за посл.месяц, т'].ravel()
    dates = df_initial['Дата'].ravel()

    unique_wells = list(set(wells))

    wells_data_dict = {}

    for well in unique_wells:
        wells_data_dict[well] = [[], [], [], []]
    for i, well in enumerate(wells):
        wells_data_dict[well][0] += [last_oil_rate[i]]
        wells_data_dict[well][1] += [last_fluid_rate[i]]
        wells_data_dict[well][2] += [time_in_production[i]]
        wells_data_dict[well][3] += [dates[i]]

    return oiz_dict, boundaries_dict, wells_data_dict


class MainWindow(QWidget):
    def __init__(
        self
    ):
        super().__init__()

        self.setWindowTitle('Статистика')
        self.setFixedSize(QSize(600, 200))

        calculate_button = QPushButton('Рассчитать')
        calculate_button.clicked.connect(choose_file_with_monthly_operating_report)

        download_button = QPushButton('Загрузить')

        tool_description_label = QLabel(
            'Инструмент для прогнозирования показателей \nбазовой добычи нефти'
            ' и обводнённости на основе \nмесячного эксплуатационного рапорта (МЭР)'
        )

        grid_box = QGridLayout()
        grid_box.addWidget(calculate_button, 0, 0)
        grid_box.addWidget(download_button, 0, 1)
        grid_box.addWidget(tool_description_label, 1, 0, 1, 2)

        self.setLayout(grid_box)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
