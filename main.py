import sys
import os
import pandas as pd
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QLabel, QFileDialog
from wells_history_preprocessing import history_preprocessing
from reserves_calculation import calculate_oiz_for_all_wells


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
    oiz = calculate_oiz_for_all_wells(df_initial, 2000, 1000, 5, 50).set_index('Скважина').T.to_dict('list')
    print(oiz)


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
